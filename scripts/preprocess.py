"""
This script processes C projects by modifying Makefiles to use a custom compiler wrapper,
running the `make` command, and preprocessing `.c` or `.h` files into `.i` files using `gcc`.
It includes special handling for specific projects with unique preprocessing requirements.
Fallback preprocessing is applied if `make` fails, ensuring `.i` files are generated.
"""

import os
import subprocess
import re

def process_special_projects(project_name, project_path):
    print(f"Processing special project {project_name} at {project_path}")
    
    if project_name == "Remimu":
        my_regex_h = os.path.join(project_path, "my_regex.h")        
        output_file = os.path.join(project_path, "my_regex.i")        
        try:
            cmd = ["gcc", "-E", my_regex_h, "-o", output_file]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Successfully preprocessed my_regex.h -> my_regex.i")
        except Exception as e:
            print(f"  Error processing Remimu project: {e}")
    elif project_name == "XOpt":
        xopt = os.path.join(project_path, "xopt.c")        
        xopt_i = os.path.join(project_path, "xopt.i")
        snprintf = os.path.join(project_path, "snprintf.c")
        snprintf_i = os.path.join(project_path, "snprintf.i")
        try:
            cmd = ["gcc", "-E", xopt, "-o", xopt_i]
            cmd_snprintf = ["gcc", "-DHAVE_STDARG_H=1", "-E", snprintf, "-o", snprintf_i]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            result_snprintf = subprocess.run(cmd_snprintf, check=True, capture_output=True, text=True)
            print(f"  Successfully preprocessed xopt.c -> xopt.i and snprintf.c -> snprintf.i")
        except Exception as e:
            print(f"  Error processing XOpt project: {e}")
    
    elif project_name == "libbase122":
        base122_c = os.path.join(project_path, "src", "base122.c")        
        output_file = os.path.join(project_path, "src", "base122.i")
        try:
            cmd = ["gcc", "-E", base122_c, "-o", output_file]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Successfully preprocessed src/base122.c -> src/base122.i")
        except Exception as e:
            print(f"  Error processing libbase122 project: {e}")
    
    elif project_name == "fleur":
        files_to_process = [
            {
                "source": os.path.join(project_path, "myutils", "myutils.c"),
                "output": os.path.join(project_path, "myutils", "myutils.i"),
                "includes": []
            },
            {
                "source": os.path.join(project_path, "fnv", "fnv.c"),
                "output": os.path.join(project_path, "fnv", "fnv.i"),
                "includes": []
            },
            {
                "source": os.path.join(project_path, "fleur", "fleur.c"),
                "output": os.path.join(project_path, "fleur", "fleur.i"),
                "includes": [os.path.join(project_path, "fnv")]
            }
        ]
        
        success_count = 0
        for file_info in files_to_process:
            source_file = file_info["source"]
            output_file = file_info["output"]
            includes = file_info["includes"]
            try:
                cmd = ["gcc", "-E", source_file, "-o", output_file]
                
                for include_path in includes:
                    cmd.append(f"-I{include_path}")
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  Successfully preprocessed {os.path.relpath(source_file, project_path)} -> {os.path.relpath(output_file, project_path)}")
                success_count += 1
            except Exception as e:
                print(f"  Error processing {os.path.relpath(source_file, project_path)}: {e}")
        
        if success_count == len(files_to_process):
            print(f"  All {success_count} files preprocessed successfully for fleur")
        elif success_count > 0:
            print(f"  Partial success: {success_count}/{len(files_to_process)} files preprocessed for fleur")
        else:
            print(f"  Failed to preprocess any files for fleur")
    
    elif project_name == "2DPartInt":
        include_path = os.path.join(project_path, "src", "include")
        files_to_process = [
            {
                "source": os.path.join(project_path, "src", "functions.c"),
                "output": os.path.join(project_path, "src", "functions.i"),
                "includes": [include_path]
            },
            {
                "source": os.path.join(project_path, "src", "collisions.c"),
                "output": os.path.join(project_path, "src", "collisions.i"),
                "includes": [include_path]
            }
        ]
        
        success_count = 0
        for file_info in files_to_process:
            source_file = file_info["source"]
            output_file = file_info["output"]
            includes = file_info["includes"]
            try:
                cmd = ["gcc", "-E", source_file, "-o", output_file]
                
                for include_path_item in includes:
                    cmd.append(f"-I{include_path_item}")
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  Successfully preprocessed {os.path.relpath(source_file, project_path)} -> {os.path.relpath(output_file, project_path)}")
                success_count += 1
            except Exception as e:
                print(f"  Error processing {os.path.relpath(source_file, project_path)}: {e}")
        
        if success_count == len(files_to_process):
            print(f"  All {success_count} files preprocessed successfully for 2DPartInt")
        elif success_count > 0:
            print(f"  Partial success: {success_count}/{len(files_to_process)} files preprocessed for 2DPartInt")
        else:
            print(f"  Failed to preprocess any files for 2DPartInt")
    
    elif project_name == "simple_lang":
        include_path = os.path.join(project_path, "Include")
        source_files = ["ast.c", "compiler.c", "lexer.c", "misc.c", "parser.c", "utils.c", "vm.c"]
        
        files_to_process = []
        for filename in source_files:
            files_to_process.append({
                "source": os.path.join(project_path, "LangKernel", filename),
                "output": os.path.join(project_path, "LangKernel", filename.replace(".c", ".i")),
                "includes": [include_path]
            })
        
        success_count = 0
        for file_info in files_to_process:
            source_file = file_info["source"]
            output_file = file_info["output"]
            includes = file_info["includes"]
            try:
                cmd = ["gcc", "-E", source_file, "-o", output_file]
                
                for include_path_item in includes:
                    cmd.append(f"-I{include_path_item}")
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  Successfully preprocessed {os.path.relpath(source_file, project_path)} -> {os.path.relpath(output_file, project_path)}")
                success_count += 1
            except Exception as e:
                print(f"  Error processing {os.path.relpath(source_file, project_path)}: {e}")
        
        if success_count == len(files_to_process):
            print(f"  All {success_count} files preprocessed successfully for simple_lang")
        elif success_count > 0:
            print(f"  Partial success: {success_count}/{len(files_to_process)} files preprocessed for simple_lang")
        else:
            print(f"  Failed to preprocess any files for simple_lang")
    
    elif project_name == "gfc":
        gfc_c = os.path.join(project_path, "src", "gfc.c")        
        try:
            with open(gfc_c, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            modified_content = content.replace('#include <gfc/gfc.h>', '#include "gfc.h"')
            with open(gfc_c, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"  Modified src/gfc.c: replaced #include <gfc/gfc.h> with #include \"gfc.h\"")
        except Exception as e:
            print(f"  Error modifying src/gfc.c: {e}")
        
        output_file = os.path.join(project_path, "src", "gfc.i")
        
        try:
            cmd = ["gcc", "-E", gfc_c, "-o", output_file]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Successfully preprocessed src/gfc.c -> src/gfc.i")
        except Exception as e:
            print(f"  Error processing gfc project: {e}")
            return False
    
    elif project_name == "Genetic-neural-network-for-simple-control":
        toolbox_path = os.path.join(project_path, "src", "toolbox")
        include_path = os.path.join(project_path, "include", "toolbox")
        subdirs = [d for d in os.listdir(toolbox_path) if os.path.isdir(os.path.join(toolbox_path, d))]
        
        files_to_process = []
        
        for subdir in subdirs:
            subdir_path = os.path.join(toolbox_path, subdir)
            files = [f for f in os.listdir(subdir_path) if f.endswith(".c")]
            for c_file in files:
                source_file = os.path.join(subdir_path, c_file)
                output_file = os.path.join(subdir_path, c_file.replace(".c", ".i"))
                files_to_process.append({
                    "source": source_file,
                    "output": output_file,
                    "includes": [include_path]
                })
        
        success_count = 0
        for file_info in files_to_process:
            source_file = file_info["source"]
            output_file = file_info["output"]
            includes = file_info["includes"]
            try:
                cmd = ["gcc", "-E", source_file, "-o", output_file]
                
                for include_path_item in includes:
                    cmd.append(f"-I{include_path_item}")
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  Successfully preprocessed {os.path.relpath(source_file, project_path)} -> {os.path.relpath(output_file, project_path)}")
                success_count += 1
            except Exception as e:
                print(f"  Error processing {os.path.relpath(source_file, project_path)}: {e}")
        
        if success_count == len(files_to_process):
            print(f"  All {success_count} files preprocessed successfully for Genetic-neural-network-for-simple-control")
        elif success_count > 0:
            print(f"  Partial success: {success_count}/{len(files_to_process)} files preprocessed for Genetic-neural-network-for-simple-control")
        else:
            print(f"  Failed to preprocess any files for Genetic-neural-network-for-simple-control")
        
    elif project_name == "clog":
        clog_h = os.path.join(project_path, "clog.h")
        output_file = os.path.join(project_path, "clog.i")
        try:
            cmd = ["gcc", "-DCLOG_MAIN", "-E", clog_h, "-o", output_file]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Successfully preprocessed clog.h -> clog.i")
        except Exception as e:
            print(f"  Error processing clog project: {e}")

    elif project_name == "libtinyfseq":
        tinyfseq_h = os.path.join(project_path, "tinyfseq.h")
        output_file = os.path.join(project_path, "tinyfseq.i")
        try:
            cmd = ["gcc", "-DTINYFSEQ_IMPLEMENTATION", "-E", tinyfseq_h, "-o", output_file]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Successfully preprocessed tinyfseq.h -> tinyfseq.i")
        except Exception as e:
            print(f"  Error processing libtinyfseq project: {e}")

    else:
        print(f"  Unknown special project: {project_name}")
            
def check_project_with_i_files(project_path):
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".i"):
                return True
    
    return False

def process_c_projects(base_dir):
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return

    projects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    wrapper_path = "../../../../clang-wrapper.sh"

    for project in projects:
        project_path = os.path.join(base_dir, project)
        
        print(f"Processing project: {project}")

        # special projects
        if project in ["Remimu", "XOpt", "libbase122", "fleur", "2DPartInt", "simple_lang", "gfc", "Genetic-neural-network-for-simple-control", "clog", "libtinyfseq"]:
            process_special_projects(project, project_path)
            continue

        makefile_name = None
        if os.path.exists(os.path.join(project_path, "Makefile")):
            makefile_name = "Makefile"
        elif os.path.exists(os.path.join(project_path, "makefile")):
            makefile_name = "makefile"
        
        # check makefile, if not found, run cmake to generate makefile
        if not makefile_name:
            print(f"  No Makefile found. Running cmake...")
            try:
                result = subprocess.run(
                    ["cmake", "."], 
                    cwd=project_path, 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                if result.returncode == 0:
                    print(f"  CMake successful.")
                    if os.path.exists(os.path.join(project_path, "Makefile")):
                        makefile_name = "Makefile"
                    elif os.path.exists(os.path.join(project_path, "makefile")):
                        makefile_name = "makefile"
                else:
                    print(f"  CMake failed for {project}")
            except Exception as e:
                print(f"  Error running cmake in {project}: {e}")
                raise e

        if makefile_name and project not in ["clog", "cset", "rubiksolver"]:
            makefile_path = os.path.join(project_path, makefile_name)

            # replace CC in makefile with wrapper path
            with open(makefile_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            new_lines = []
            cc_found = False
            cc_pattern = re.compile(r'^\s*CC\s*[:?+]?=')

            for line in lines:
                if cc_pattern.match(line):
                    new_lines.append(f"CC = {wrapper_path}\n")
                    cc_found = True
                else:
                    new_lines.append(line)
            
            if not cc_found:
                new_lines.insert(0, f"CC = {wrapper_path}\n")

            with open(makefile_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            print(f"  {makefile_name} updated for {project}")

            # run make command
            try:
                print(f"  Running make in {project_path}...")
                result = subprocess.run(
                    ["make"], 
                    cwd=project_path, 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0 and check_project_with_i_files(project_path):
                    print(f"  Make successful for {project}")
                    continue
                else:
                    print(f"  Make failed for {project}")
            except subprocess.TimeoutExpired:
                print(f"  Make timed out for {project}")
            except Exception as e:
                print(f"  Error running make in {project}: {e}")
                
        print(f"  Attempting fallback preprocessing for {project}...")
        
        c_files = []
        h_files = []
        
        skip_dirs = {'test', 'bin', 'example', 'dev-deps'}
        
        for root, dirs, files in os.walk(project_path):
            rel_path = os.path.relpath(root, project_path)
            
            path_parts = rel_path.split(os.sep)
            
            should_skip = False
            for part in path_parts:
                if any(skip_kw in part.lower() for skip_kw in skip_dirs):
                    should_skip = True
                    break
            
            if should_skip:
                continue

            for file in files:
                if file.endswith(".c"):
                    c_files.append(os.path.join(root, file))
                elif file.endswith(".h"):
                    h_files.append(os.path.join(root, file))
        
        non_test_c_files = [f for f in c_files if "test" not in os.path.basename(f).lower()]
        
        files_to_process = []
        if not non_test_c_files:
            print(f"  No non-test .c files found. Processing .h files...")
            files_to_process = h_files
        else:
            print(f"  Found .c files. Processing all .c files...")
            files_to_process = non_test_c_files
        
        success_count = 0
        for src_file in files_to_process:
            output_file = os.path.splitext(src_file)[0] + ".i"
            
            try:
                cmd = ["gcc", "-E", src_file, "-o", output_file, f"-I{project_path}"]
                
                include_dir = os.path.join(project_path, "include")
                if os.path.exists(include_dir):
                    cmd.append(f"-I{include_dir}")
                    
                src_dir = os.path.join(project_path, "src")
                if os.path.exists(src_dir):
                    cmd.append(f"-I{src_dir}")

                subprocess.run(cmd, check=True, capture_output=True)
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"    Failed to preprocess {os.path.basename(src_file)}")
            except Exception as e:
                print(f"    Error processing {os.path.basename(src_file)}: {e}")
        
        print(f"  Fallback preprocessing completed. {success_count}/{len(files_to_process)} files processed.")

if __name__ == "__main__":
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    cbench_dir = os.path.join(root_dir, "data/project/CBench")
    
    process_c_projects(cbench_dir)
    