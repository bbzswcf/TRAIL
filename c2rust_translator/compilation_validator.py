"""
编译验证和修复模块, 负责编译Rust代码并在失败时进行修复
"""

import os
import re
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_COMPILE_TIMEOUT_SECONDS = 180


class CompilationValidator:
    """编译验证器"""
    
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
    
    def compile_project(
        self,
        check_only: bool = True,
        include_tests: bool = False,
        timeout: int = DEFAULT_COMPILE_TIMEOUT_SECONDS,
    ) -> Tuple[bool, str]:
        cmd = ['cargo', 'check' if check_only else 'build']
        if include_tests:
            cmd.append('--tests')
        new_env = os.environ.copy()
        new_env["RUSTFLAGS"] = "-Awarnings"
        
        try:
            result = subprocess.Popen(
                cmd,
                cwd=self.project_dir,
                env=new_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            stdout, stderr = result.communicate(timeout=timeout)
            
            success = result.returncode == 0
            output = stdout + stderr
            
            return success, output
        
        except subprocess.TimeoutExpired:
            _kill_process_group(result)
            stdout, stderr = result.communicate()
            output = (stdout or "") + (stderr or "")
            if output:
                return False, f"Compilation timeout after {timeout} seconds\n{output}"
            return False, f"Compilation timeout after {timeout} seconds"
        except Exception as e:
            return False, f"Compilation error: {e}"
    
    def parse_compilation_errors(self, output: str) -> list:
        """
        解析编译错误信息
        """
        errors = []
        
        # 匹配错误行
        error_pattern = re.compile(r'error(?:\[E\d+\])?: (.*?)(?=\n\s*-->|\nerror|\nwarning|\n\n|$)', re.DOTALL)
        matches = error_pattern.finditer(output)
        
        for match in matches:
            errors.append(match.group(1).strip())
        
        return errors

    def parse_compilation_error_details(self, output: str) -> List[Dict]:
        """
        结构化解析编译错误，返回：
        - code: 错误码（如 E0308）
        - message: 错误消息首行
        - file/line/column: 位置（若能解析）
        - location: `file:line:column` 字符串（若能解析）
        - excerpt: 错误块摘录
        """
        details: List[Dict] = []
        lines = (output or "").splitlines()
        i = 0
        n = len(lines)

        error_head_re = re.compile(r'^error(?:\[(E\d+)\])?:\s*(.*)$')
        loc_re = re.compile(r'^\s*-->\s+(.+?):(\d+):(\d+)\s*$')

        while i < n:
            head = lines[i]
            m = error_head_re.match(head)
            if not m:
                i += 1
                continue

            code = m.group(1)
            message = (m.group(2) or "").strip()
            block_lines = [head]

            file_path = None
            line_no = None
            col_no = None

            j = i + 1
            while j < n:
                next_line = lines[j]
                if error_head_re.match(next_line):
                    break
                block_lines.append(next_line)
                if file_path is None:
                    lm = loc_re.match(next_line)
                    if lm:
                        file_path = lm.group(1).strip()
                        try:
                            line_no = int(lm.group(2))
                            col_no = int(lm.group(3))
                        except Exception:
                            line_no = None
                            col_no = None
                j += 1

            entry: Dict = {
                "code": code,
                "message": message,
                "file": file_path,
                "line": line_no,
                "column": col_no,
                "location": f"{file_path}:{line_no}:{col_no}" if (file_path and line_no and col_no) else None,
                "excerpt": "\n".join(block_lines[:30]).strip(),
            }
            details.append(entry)
            i = j

        return details


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception:
        process.kill()
