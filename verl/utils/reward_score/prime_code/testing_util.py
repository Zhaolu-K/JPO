# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import faulthandler
import json
import platform

# to run the solution files we're using a timing based approach
import signal
import sys
import traceback

# used for debugging to time steps
from datetime import datetime
from enum import Enum

# for capturing the stdout
from io import StringIO

# used for testing the code that reads from input
from unittest.mock import mock_open, patch

import numpy as np
from pyext import RuntimeModule


def truncatefn(s, length=300):
    assert isinstance(s, str)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1

# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def only_int_check(val):
    return isinstance(val, int)


def string_int_check(val):
    return isinstance(val, str) and val.isdigit()


def combined_int_check(val):
    return only_int_check(val) or string_int_check(val)


def clean_traceback(error_traceback):
    file_start = error_traceback.find('File "<string>"')
    # print(file_start)
    error_traceback = "Traceback (most recent call last):\n  " + error_traceback[file_start:]
    return error_traceback


_SOL_PREAMBLE = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n"  # noqa: E501


class _CaseAction(Enum):
    """标准输入比较级联的三种退出方式，对应原实现的三类控制流。"""

    passed = 0  # 记下结果后跳到下一个用例
    skipped = 1  # 不记结果直接跳过（check4 的 except 分支）
    finished = 2  # 记下结果；若不为 True 则以 Wrong Answer 返回


def _resolve_code_type(in_outs):
    """判断是调用式还是标准输入式，并给出待取的方法名。"""
    which_type = None
    method_name = None
    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs["fn_name"]
    return which_type, method_name


def _strip_main_guard(test):
    """if code has if __name__ == "__main__": then remove it"""
    try:
        astree = ast.parse(test)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                test = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    return test


def _wrap_as_code_function(test):
    """把标准输入式解答缩进包进 `def code():` 里。"""
    tmp_test = test.split("\n")

    new_test = []
    for x in tmp_test:
        if (not x.startswith("from ")) and (not x.startswith("import ")):
            new_test.append("\t" + x + "\n")
        else:
            new_test.append(x + "\n")
    tmp_test = new_test

    new_test = ""
    started = False
    for i in tmp_test:
        if i.startswith("\t") and not started:
            new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
            new_test += "def code():\n"
            new_test += i
            started = True
        elif started and ((i.startswith("from ")) or (i.startswith("import "))):
            new_test += "\t" + i
        else:
            new_test += i
    return new_test


def _compilation_error(e, error_traceback):
    return {
        "error": repr(e),
        # "error_code": -1,
        # "error_message": "Compilation Error",
        "traceback": clean_traceback(error_traceback),
    }


def _compile_call_based(sol, test, timeout, debug):
    """编译调用式解答，返回 (模块或实例, 错误响应)。"""
    if debug:
        print(f"sol = {sol}")
    signal.alarm(timeout)
    try:
        tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
        tmp = tmp_sol if "class Solution" not in test else tmp_sol.Solution()
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        error_traceback = traceback.format_exc()
        if debug:
            print(f"type 0 compilation error = {e}")
        return None, _compilation_error(e, error_traceback)
    signal.alarm(0)
    return tmp, None


def _compile_standard_input(sol, timeout, debug):
    """编译标准输入式解答，返回 (模块, 错误响应)。"""
    if debug:
        print(f"sol = {sol}")
    signal.alarm(timeout)
    try:
        tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
        tmp = tmp_sol
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        error_traceback = traceback.format_exc()
        if debug:
            print(f"type 1 compilation error = {e}")
        return None, _compilation_error(e, error_traceback)
    signal.alarm(0)
    return tmp, None


def _truncate_case_io(in_outs, index, inputs, raw_inputs, raw_outputs, which_type):
    """按用例类型解析输入并截断用于报错展示的原文。"""
    if which_type == CODE_TYPE.call_based:
        inputs = [json.loads(line) for line in inputs.split("\n")]
        in_outs["outputs"][index] = json.loads(in_outs["outputs"][index])

        truncate_line_size = 300 // (raw_inputs.count("\n") + 1)
        raw_inputs = "\n".join([truncatefn(line, truncate_line_size) for line in raw_inputs.strip().split("\n")])
        raw_outputs = truncatefn(raw_outputs, 200)
    else:
        raw_inputs = truncatefn(raw_inputs)
        raw_outputs = truncatefn(raw_outputs, 200)
    return inputs, raw_inputs, raw_outputs


def _undo_json_string_keys(inputs, in_outs, index):
    """JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)"""
    try:
        if isinstance(inputs[0], dict):
            inputs = [{int(k): v for k, v in inputs[0].items()}]
    except Exception:
        pass
    try:
        if isinstance(in_outs["outputs"][index], dict):
            in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index].items()}]
    except Exception:
        pass
    try:
        if isinstance(in_outs["outputs"][index][0], dict):
            in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index][0].items()}]
    except Exception:
        pass
    return inputs


def _compare_call_based(output, in_outs, index):
    tmp_result = output == in_outs["outputs"][index]
    if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
        tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

    # ground truth sequences are not tuples
    try:
        if isinstance(output[0], tuple):
            tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
    except Exception:
        pass
    return tmp_result


def _split_expected_lines(in_outs, index):
    """try one more time without \\n"""
    if isinstance(in_outs["outputs"][index], list):
        for tmp_index, i in enumerate(in_outs["outputs"][index]):
            in_outs["outputs"][index][tmp_index] = i.split("\n")
            in_outs["outputs"][index][tmp_index] = [x.strip() for x in in_outs["outputs"][index][tmp_index] if x]
    else:
        in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
        in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
        in_outs["outputs"][index] = list(map(lambda x: x.strip(), in_outs["outputs"][index]))


def _split_expected_words(in_outs, index):
    """try by converting the stuff into split up list"""
    if isinstance(in_outs["outputs"][index], list):
        for tmp_index, i in enumerate(in_outs["outputs"][index]):
            in_outs["outputs"][index][tmp_index] = set(i.split())
    else:
        in_outs["outputs"][index] = set(in_outs["outputs"][index].split())


def _split_output_words(output):
    """try by converting the output into a split up list too"""
    if isinstance(output, list):
        for tmp_index, i in enumerate(output):
            output[tmp_index] = i.split()
        output = list(filter(len, output))
        for tmp_index, i in enumerate(output):
            output[tmp_index] = set(i)
    else:
        output = output.split()
        output = list(filter(len, output))
        output = set(output)
    return output


def _compare_lists(output, in_outs, index, tmp_result, label, debug):
    try:
        tmp_result = output == [in_outs["outputs"][index]]
        if isinstance(in_outs["outputs"][index], list):
            tmp_result = tmp_result or (output == in_outs["outputs"][index])
    except Exception as e:
        if debug:
            print(f"Failed {label} exception = {e}")
        pass
    return tmp_result


def _compare_as_floats_flat(output, in_outs, index, tmp_result, debug):
    try:
        all_ints = all(combined_int_check(e1) and combined_int_check(e2) for e1, e2 in zip(output, in_outs["outputs"][index]))
        if not all_ints:
            if debug:
                print([combined_int_check(e1) and combined_int_check(e2) for e1, e2 in zip(output, in_outs["outputs"][index])])
            output_float = [float(e) for e in output]
            gt_float = [float(e) for e in in_outs["outputs"][index]]
            tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
    except Exception:
        pass
    return tmp_result


def _compare_as_floats_nested(output, in_outs, index, tmp_result):
    try:
        if isinstance(output[0], list):
            all_ints = all(combined_int_check(e1) and combined_int_check(e2) for e1, e2 in zip(output[0], in_outs["outputs"][index]))
            if not all_ints:
                output_float = [float(e) for e in output[0]]
                gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                tmp_result = tmp_result or ((len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
    except Exception:
        pass
    return tmp_result


def _debug_dump(label, output, in_outs, index, inputs, tmp_result=None, show_result=False):
    nl = "\n"
    suffix = f" {tmp_result=}" if show_result else ""
    if not isinstance(inputs, list):
        print(f"{label} output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}{suffix}")
    else:
        print(f"{label} output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}{suffix}")


def _compare_std_exact(output, in_outs, index, debug):
    """级联前半段：原样、按行、去空白比对。返回 (是否已定论, 结果, output)。"""
    if custom_compare_(output, in_outs["outputs"][index]):
        return True, True, output

    # ground truth sequences are expressed as lists not tuples
    if isinstance(output, tuple):
        output = list(output)

    tmp_result = False
    try:
        tmp_result = output == [in_outs["outputs"][index]]
        if isinstance(in_outs["outputs"][index], list):
            tmp_result = tmp_result or (output == in_outs["outputs"][index])
            if isinstance(output[0], str):
                tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
    except Exception as e:
        if debug:
            print(f"Failed check1 exception = {e}")
        pass

    if tmp_result is True:
        return True, tmp_result, output

    _split_expected_lines(in_outs, index)
    tmp_result = _compare_lists(output, in_outs, index, tmp_result, "check2", debug)

    return tmp_result is True, tmp_result, output


def _compare_std_relaxed(output, in_outs, index, inputs, tmp_result, debug):
    """级联后半段：过滤空行、浮点近似、按词集合比对。"""
    # try by converting the output into a split up list too
    if isinstance(output, list):
        output = list(filter(len, output))

    if debug:
        _debug_dump("@1", output, in_outs, index, inputs, tmp_result, show_result=True)
        print(f"{tmp_result=} @a")

    tmp_result = _compare_lists(output, in_outs, index, tmp_result, "check3", debug)

    if debug:
        print(f"{tmp_result=} @b")

    tmp_result = _compare_as_floats_flat(output, in_outs, index, tmp_result, debug)
    tmp_result = _compare_as_floats_nested(output, in_outs, index, tmp_result)

    if debug:
        print(f"{tmp_result=} @c")

    if tmp_result is True:
        return _CaseAction.passed, tmp_result, output

    if debug:
        print(f"{tmp_result=} @d")
    _split_expected_words(in_outs, index)

    if debug:
        print(f"{tmp_result=} @e")

    try:
        tmp_result = output == in_outs["outputs"][index]
    except Exception as e:
        if debug:
            print(f"Failed check4 exception = {e}")
        return _CaseAction.skipped, tmp_result, output

    if tmp_result is True:
        return _CaseAction.passed, tmp_result, output

    if debug:
        print(f"{tmp_result=} @f")

    output = _split_output_words(output)

    if debug:
        print(f"{tmp_result=} @g")

    if tmp_result is True and debug:
        print("PASSED")

    return _CaseAction.finished, tmp_result, output


def _compare_standard_output(output, in_outs, index, inputs, debug):
    """逐级降级比对标准输出，返回 (动作, 结果, 最终 output)。

    级联会就地修改 in_outs["outputs"][index]，与原实现一致。
    """
    settled, tmp_result, output = _compare_std_exact(output, in_outs, index, debug)
    if settled:
        return _CaseAction.passed, tmp_result, output
    return _compare_std_relaxed(output, in_outs, index, inputs, tmp_result, debug)


def _wrong_answer(raw_true_output_copy, raw_outputs, raw_inputs):
    return {
        "output": raw_true_output_copy,
        "expected": raw_outputs,
        "inputs": raw_inputs,
        # "error_code": -2,
        "error_message": "Wrong Answer",
    }


def _run_call_based_case(method, inputs, in_outs, index, raw_inputs, raw_outputs, timeout, debug):
    """跑一个调用式用例，返回 (要记入 results 的值, 终止响应或 None)。"""
    signal.alarm(timeout)
    faulthandler.enable()
    try:
        output = method(*inputs)
        raw_true_output = output  # noqa: F841

        raw_true_output_copy = json.dumps(output)
        raw_true_output_copy = truncatefn(raw_true_output_copy, 200)

        # ground truth sequences are not tuples
        if isinstance(output, tuple):
            output = list(output)

        tmp_result = _compare_call_based(output, in_outs, index)
        if tmp_result is not True:
            return tmp_result, _wrong_answer(raw_true_output_copy, raw_outputs, raw_inputs)
        # reset the alarm
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        error_traceback = traceback.format_exc()
        faulthandler.disable()
        if debug:
            print(f"Standard input runtime error or time limit exceeded error = {e}")
        return -1, {
            "error": repr(e),
            "traceback": clean_traceback(error_traceback),
        }
    faulthandler.disable()
    signal.alarm(0)
    if debug:
        print(f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
    return tmp_result, None


def _capture_standard_output(method, inputs, timeout):
    """在捕获 stdout 的前提下执行解答，返回 (输出行, 截断副本, passed, 错误响应或 None)。"""
    passed = False
    signal.alarm(timeout)
    with Capturing() as output:
        try:
            call_method(method, inputs)
            # reset the alarm
            signal.alarm(0)
            passed = True
        except Exception as e:
            # runtime error or took too long
            signal.alarm(0)
            error_traceback = traceback.format_exc()
            print(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
            return None, None, False, {
                "error": repr(e),
                "traceback": clean_traceback(error_traceback),
            }
        signal.alarm(0)
    raw_true_output = output[0]
    raw_true_output_copy = truncatefn(raw_true_output, 200)
    return raw_true_output.splitlines(), raw_true_output_copy, passed, None


def _run_standard_input_case(method, inputs, in_outs, index, raw_inputs, raw_outputs, timeout, debug):
    """跑一个标准输入用例，返回 (动作, 要记入 results 的值, 终止响应或 None)。"""
    faulthandler.enable()

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)
    if isinstance(in_outs["outputs"][index], list):
        in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

    output, raw_true_output_copy, passed, error = _capture_standard_output(method, inputs, timeout)
    if error is not None:
        return _CaseAction.finished, -1, error

    if not passed:
        if debug:
            _debug_dump("not passed", output, in_outs, index, inputs)
        return _CaseAction.skipped, None, None

    if passed and debug:
        print(f"==> output = {output}, test outputs = {in_outs['outputs'][index]}")

    action, tmp_result, output = _compare_standard_output(output, in_outs, index, inputs, debug)
    if action is _CaseAction.skipped:
        return _CaseAction.skipped, None, None
    if action is _CaseAction.passed:
        return _CaseAction.passed, tmp_result, None

    if tmp_result is not True:
        return _CaseAction.finished, tmp_result, _wrong_answer(raw_true_output_copy, raw_outputs, raw_inputs)

    if debug:
        _debug_dump("@2", output, in_outs, index, inputs)
    return _CaseAction.passed, tmp_result, None


def _prepare_solution(in_outs, test, timeout, debug):
    """编译解答并取出待调用的方法，返回 (method, which_type, 错误值, 错误响应)。"""
    which_type, method_name = _resolve_code_type(in_outs)

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")

    if test is None:
        raise AssertionError("should not happen: test code is none")

    sol = _SOL_PREAMBLE
    if debug:
        print(f"loading test code = {datetime.now().time()}")

    tmp = None
    if which_type == CODE_TYPE.call_based:
        sol += test
        tmp, error = _compile_call_based(sol, test, timeout, debug)
        if error is not None:
            return None, which_type, -2, error
    elif which_type == CODE_TYPE.standard_input:
        test = _strip_main_guard(test)
        sol += _wrap_as_code_function(test)
        method_name = "code"
        tmp, error = _compile_standard_input(sol, timeout, debug)
        if error is not None:
            return None, which_type, -2, error

    if debug:
        print(f"get method = {datetime.now().time()}")

    try:
        method = getattr(tmp, method_name)  # get_attr second arg must be str
    except Exception:
        signal.alarm(0)
        error_traceback = traceback.format_exc()
        error_info = sys.exc_info()
        print(f"unable to get function error = {error_info}")
        return None, which_type, -2, {
            "error": repr(error_info),
            # "error_code": -1,
            # "error_message": "Unable to extract code",
            "traceback": clean_traceback(error_traceback),
        }
    return method, which_type, None, None


def run_test(in_outs, test=None, debug=False, timeout=15):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")

    results = []
    method, which_type, failure, error = _prepare_solution(in_outs, test, timeout, debug)
    if error is not None:
        results.append(failure)
        return results, error

    for index, inputs in enumerate(in_outs["inputs"]):
        raw_inputs = inputs
        raw_outputs = in_outs["outputs"][index]
        inputs, raw_inputs, raw_outputs = _truncate_case_io(
            in_outs, index, inputs, raw_inputs, raw_outputs, which_type
        )
        inputs = _undo_json_string_keys(inputs, in_outs, index)

        if debug:
            print(f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}")

        if which_type == CODE_TYPE.call_based:  # Call-based
            value, response = _run_call_based_case(
                method, inputs, in_outs, index, raw_inputs, raw_outputs, timeout, debug)
            results.append(value)
            if response is not None:
                return results, response
        elif which_type == CODE_TYPE.standard_input:  # Standard input
            action, value, response = _run_standard_input_case(
                method, inputs, in_outs, index, raw_inputs, raw_outputs, timeout, debug)
            if action is _CaseAction.skipped:
                continue
            results.append(value)
            if response is not None:
                return results, response
            if debug:
                print(f"results = {results}")

    return results, {}


def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None  # 防止干扰repl评测
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
