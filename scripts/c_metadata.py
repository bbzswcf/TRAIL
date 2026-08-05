import os
import json
import re
import subprocess
from tree_sitter import Language, Parser
import tree_sitter_c

# 初始化 Tree-sitter
C_LANGUAGE = Language(tree_sitter_c.language())
parser = Parser(C_LANGUAGE)

def _contains_function_declarator(node):
    if node.type == 'function_declarator':
        return True
    for child in node.children:
        if _contains_function_declarator(child):
            return True
    return False

def preprocess_i_file_content(content):
    """
    预处理 .i 文件内容：
    1. 删除空行
    2. 如果行号信息以 # 开头且来自 /usr/，删除后续代码直到下一个行号信息
    3. 删除所有行号信息行
    """
    processed_lines = []
    keep_code = True

    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith('#'):
            keep_code = "\"/usr/" not in stripped
            continue
        if not stripped:
            continue
        if keep_code:
            processed_lines.append(line)

    return '\n'.join(processed_lines)

def format_with_clang(content, temp_file_path):
    """使用 clang-format 格式化代码"""
    try:
        # 运行 clang-format
        result = subprocess.run(
            ['clang-format', '-style=Microsoft'],
            input=content.encode('utf-8'),
            capture_output=True,
            timeout=30
        )

        if result.returncode == 0:
            return result.stdout.decode('utf-8', errors='ignore')
        else:
            print(f"  Warning: clang-format failed for {temp_file_path}")
            print(f"  Error: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
            return content
    except Exception as e:
        print(f"  Warning: Error running clang-format: {e}")
        return content

def extract_function_name(node):
    """从函数定义节点提取函数名"""
    for child in node.children:
        if child.type == 'function_declarator':
            for subchild in child.children:
                if subchild.type == 'identifier':
                    return subchild.text.decode('utf-8', errors='ignore')
                elif subchild.type == 'parenthesized_declarator':
                    # 处理返回函数指针的函数，如: (*mk_num(...))(...)
                    return extract_function_name(subchild)
        elif child.type == 'pointer_declarator':
            return extract_function_name(child)
    return None

def extract_struct_name(node):
    """从结构体定义节点提取结构体名"""
    for child in node.children:
        if child.type == 'type_identifier':
            return child.text.decode('utf-8', errors='ignore')
    return None

def extract_enum_name(node):
    """从枚举定义节点提取枚举名，如果是匿名枚举，返回所有枚举值标识符"""
    # 首先检查是否有枚举名
    for child in node.children:
        if child.type == 'type_identifier':
            return [child.text.decode('utf-8', errors='ignore')]

    # 如果没有枚举名（匿名枚举），提取所有枚举值
    enum_values = []
    for child in node.children:
        if child.type == 'enumerator_list':
            for enum_item in child.children:
                if enum_item.type == 'enumerator':
                    # 从 enumerator 中提取标识符
                    for subchild in enum_item.children:
                        if subchild.type == 'identifier':
                            enum_values.append(subchild.text.decode('utf-8', errors='ignore'))
                            break

    return enum_values if enum_values else None

def extract_typedef_name(node):
    """从 typedef 节点提取类型别名，包括函数指针类型和数组类型"""
    def search_typedef_alias_in_declarator(n):
        """优先从 declarator 里提取 typedef 别名，避免拿到 struct/union/enum tag"""
        if n.type in ['type_identifier', 'identifier']:
            return n.text.decode('utf-8', errors='ignore')

        # 这些节点里可能包含真正的 typedef 别名
        declarator_like_nodes = {
            'type_definition',
            'pointer_declarator',
            'function_declarator',
            'array_declarator',
            'parenthesized_declarator',
            'attributed_declarator',
        }
        if n.type not in declarator_like_nodes:
            return None

        # 从右向左找，typedef 别名通常在声明末尾
        for child in reversed(n.children):
            # 跳过函数参数，避免把参数名当成 typedef 名
            if child.type == 'parameter_list':
                continue
            alias = search_typedef_alias_in_declarator(child)
            if alias:
                return alias
        return None

    def search_for_name(n):
        """递归搜索类型名称"""
        if n.type in ['type_identifier', 'identifier']:
            return n.text.decode('utf-8', errors='ignore')

        for child in n.children:
            if child.type == 'array_declarator':
                # 处理数组类型 typedef
                for subchild in child.children:
                    if subchild.type == 'type_identifier':
                        return subchild.text.decode('utf-8', errors='ignore')
            elif child.type == 'function_declarator':
                # 在函数声明中查找 parenthesized_declarator
                for subchild in child.children:
                    if subchild.type == 'parenthesized_declarator':
                        # 在括号内查找 pointer_declarator
                        for sub_subchild in subchild.children:
                            if sub_subchild.type == 'pointer_declarator':
                                # 在 pointer_declarator 中查找 type_identifier
                                for identifier_node in sub_subchild.children:
                                    if identifier_node.type == 'type_identifier':
                                        return identifier_node.text.decode('utf-8', errors='ignore')
            elif child.type == 'pointer_declarator':
                # 处理普通指针类型或嵌套的函数指针
                # 先检查是否有嵌套的 function_declarator
                for subchild in child.children:
                    if subchild.type == 'function_declarator':
                        # 递归搜索嵌套的函数声明
                        name = search_for_name(child)
                        if name:
                            return name
                    elif subchild.type in ['type_identifier', 'identifier']:
                        return subchild.text.decode('utf-8', errors='ignore')
            elif child.type in ['type_identifier', 'identifier']:
                return child.text.decode('utf-8', errors='ignore')

        return None

    # 先从 declarator 中提取 typedef 别名，避免返回 struct tag
    alias = search_typedef_alias_in_declarator(node)
    if alias:
        return alias

    # 再尝试原有递归搜索逻辑
    name = search_for_name(node)
    if name:
        return name

    # 如果递归搜索没找到，检查是否有简单的 primitive_type（如 typedef short bool;）
    for child in node.children:
        if child.type == 'primitive_type':
            return child.text.decode('utf-8', errors='ignore')

    return None

def extract_variable_name(node):
    """从声明节点提取变量名"""
    for child in node.children:
        if child.type == 'init_declarator':
            for subchild in child.children:
                if subchild.type == 'identifier':
                    return subchild.text.decode('utf-8', errors='ignore')
                elif subchild.type == 'function_declarator':
                    # 处理函数指针数组等复杂声明
                    name = extract_name_from_function_declarator(subchild)
                    if name:
                        return name
                elif subchild.type in ['pointer_declarator', 'array_declarator']:
                    name = extract_variable_name_from_declarator(subchild)
                    if name:
                        return name
        elif child.type == 'identifier':
            return child.text.decode('utf-8', errors='ignore')
        elif child.type == 'function_declarator':
            # 处理顶层的函数声明符
            name = extract_name_from_function_declarator(child)
            if name:
                return name
        elif child.type in ['pointer_declarator', 'array_declarator']:
            name = extract_variable_name_from_declarator(child)
            if name:
                return name
    return None

def extract_name_from_function_declarator(node):
    """从 function_declarator 中提取变量名"""
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode('utf-8', errors='ignore')
        elif child.type == 'parenthesized_declarator':
            # 在括号声明中查找
            for subchild in child.children:
                if subchild.type == 'identifier':
                    return subchild.text.decode('utf-8', errors='ignore')
                elif subchild.type in ['pointer_declarator', 'array_declarator']:
                    name = extract_variable_name_from_declarator(subchild)
                    if name:
                        return name
    return None

def extract_variable_name_from_declarator(node):
    """从 declarator 节点中提取变量名"""
    for child in node.children:
        if child.type == 'identifier':
            return child.text.decode('utf-8', errors='ignore')
        elif child.type in ['pointer_declarator', 'array_declarator']:
            name = extract_variable_name_from_declarator(child)
            if name:
                return name
    return None

def analyze_i_file(file_path):
    """使用 Tree-sitter 分析 .i 文件"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # 预处理 .i 文件内容
    content_str = content.decode('utf-8', errors='ignore')
    preprocessed_content = preprocess_i_file_content(content_str)
    formatted_content = preprocessed_content
    
    # 生成临时文件路径
    base_name = os.path.splitext(file_path)[0]
    temp_file = f"{base_name}_del_row_info.i"
    processed_file = f"{base_name}_processed.txt"
    
    try:
        # 保存预处理后的内容到 xxx_del_row_info.i
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(preprocessed_content)
        
        # 使用 clang-format 格式化
        formatted_content = format_with_clang(preprocessed_content, temp_file)
        
        # 保存格式化后的内容到 xxx_processed.txt
        with open(processed_file, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        # 删除临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    except Exception as e:
        print(f"  Error during preprocessing/formatting: {e}")
        # 清理临时文件
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
    
    # 使用预处理后的内容进行 tree-sitter 分析
    content = formatted_content.encode('utf-8')

    tree = parser.parse(content)
    root = tree.root_node
    
    # 改为字典结构
    data = {
        "globals": {},
        "structs": {},
        "unions": {},
        "enums": {},
        "functions": {},
        "typedefs": {}
    }
    
    # 用于生成唯一的匿名名称
    counters = {
        "globals": 0,
        "structs": 0,
        "unions": 0,
        "enums": 0,
        "functions": 0,
        "typedefs": 0
    }
    
    for node in root.children:
        text = node.text.decode('utf-8', errors='ignore').strip()
        if not text:
            continue

        if node.type == 'function_definition':
            # 函数定义
            func_name = extract_function_name(node)
            if not func_name:
                func_name = f'anonymous_func_{counters["functions"]}'
                counters["functions"] += 1
            data['functions'][func_name] = text
        
        elif node.type == 'struct_specifier':
            # struct定义
            struct_name = extract_struct_name(node)
            if not struct_name:
                struct_name = f'anonymous_struct_{counters["structs"]}'
                counters["structs"] += 1
            data['structs'][struct_name] = text
        
        elif node.type == 'union_specifier':
            # union定义
            union_name = extract_struct_name(node)  # union 和 struct 提取方式相同
            if not union_name:
                union_name = f'anonymous_union_{counters["unions"]}'
                counters["unions"] += 1
            data['unions'][union_name] = text
        
        elif node.type == 'enum_specifier':
            # enum定义
            enum_names = extract_enum_name(node)
            if not enum_names:
                enum_name = f'anonymous_enum_{counters["enums"]}'
                counters["enums"] += 1
                data['enums'][enum_name] = text
            else:
                # 如果返回多个名称（匿名枚举的枚举值），为每个值创建条目
                for enum_name in enum_names:
                    data['enums'][enum_name] = text
        
        elif node.type == 'type_definition':
            # typedef 声明
            typedef_name = extract_typedef_name(node)
            if not typedef_name:
                typedef_name = f'anonymous_typedef_{counters["typedefs"]}'
                counters["typedefs"] += 1
                
            # 检查 typedef 中是否包含结构体/联合体/枚举的定义体
            typedef_decl = False
            for child in node.children:
                if child.type == 'struct_specifier':
                    # typedef 场景优先使用别名作为key，避免被 struct tag 覆盖
                    data['structs'][typedef_name] = text
                    typedef_decl = True
                    break
                elif child.type == 'union_specifier':
                    # typedef 场景优先使用别名作为key，避免被 union tag 覆盖
                    data['unions'][typedef_name] = text
                    typedef_decl = True
                    break
                elif child.type == 'enum_specifier':
                    # typedef 场景优先使用别名作为key，避免被 enum tag/枚举值覆盖
                    data['enums'][typedef_name] = text
                    typedef_decl = True
                    break
                
            if not typedef_decl:
                data['typedefs'][typedef_name] = text
            
        elif node.type == 'declaration':
            # 检查是否是变量声明
            has_declarator = False
            for child in node.children:
                # 忽略存储类说明符 (static, extern 等) 和 类型限定符 (const 等)
                if child.type in [';', 'storage_class_specifier', 'type_qualifier', 'primitive_type', 
                                  'type_identifier', 'sized_type_specifier', 'struct_specifier', 
                                  'union_specifier', 'enum_specifier']:
                    continue
                
                # 如果遇到 function_declarator，说明是函数声明，跳过
                flag = False
                if child.type == 'function_declarator':
                    flag = True
                elif child.type == 'pointer_declarator':
                    for subchild in child.children:
                        if _contains_function_declarator(subchild):
                            flag = True
                            break
                if flag:
                    break

                # 如果还有其他节点，通常是 declarator
                if child.type in ['identifier', 'init_declarator', 'pointer_declarator', 
                                  'array_declarator']:
                    has_declarator = True
                    break
            
            if has_declarator:
                # 这是变量声明
                var_name = extract_variable_name(node)
                if not var_name:
                    var_name = f'anonymous_var_{counters["globals"]}'
                    counters["globals"] += 1
                data['globals'][var_name] = text

    # 如果 counters 中有非零元素，输出文件路径
    if any(count > 0 for count in counters.values()):
        print(f"File {file_path} contains anonymous elements: {counters}")

    return data

def extract_includes(file_path):
    """从源代码中提取 #include "xxx" """
    includes = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # 匹配 #include "xxx"
            matches = re.findall(r'#include\s*"([^"]+)"', content)
            includes.extend(matches)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return includes

def process_projects(base_dir, output_dir):
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist")
        return

    projects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # 定义要跳过的目录
    skip_dirs = {'test', 'bin', 'example', 'dev-deps'}
    
    for project in projects:
        print(f"Processing {project}...")
        # if project != "Genetic-neural-network-for-simple-control":
        #     continue
        project_path = os.path.join(base_dir, project)
        # 改为字典结构
        project_metadata = {
            "project_name": project,
            "files": {}
        }
        
        for root, _, files in os.walk(project_path):
            # 检查当前路径是否包含要跳过的目录
            rel_root = os.path.relpath(root, project_path)
            path_parts = set(rel_root.split(os.sep))
            if path_parts & skip_dirs:  # 如果路径中包含任何skip_dirs
                continue
            
            for file in files:
                if file.endswith(".i") and 'test' not in file.lower():
                    i_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(i_file_path, project_path)
                    
                    # 寻找同名 .c 或 .h
                    base_name = os.path.splitext(file)[0]
                    c_file_path = os.path.join(root, base_name + ".c")
                    h_file_path = os.path.join(root, base_name + ".h")
                    
                    original_file_path = None
                    includes = []
                    
                    if os.path.exists(c_file_path):
                        original_file_path = c_file_path
                    elif os.path.exists(h_file_path):
                        original_file_path = h_file_path
                    
                    if original_file_path:
                        includes = extract_includes(original_file_path)
                        original_rel_path = os.path.relpath(original_file_path, project_path)
                    else:
                        original_rel_path = None
                    
                    # 分析 .i 文件
                    analysis_data = analyze_i_file(i_file_path)
                    
                    if analysis_data:
                        file_info = {
                            "relative_path": rel_path,
                            "original_file": original_rel_path,
                            "includes": includes,
                            "declarations": analysis_data
                        }
                        # 使用 relative_path 作为键
                        project_metadata["files"][rel_path] = file_info
        
        # 保存结果
        output_file = os.path.join(output_dir, f"{project}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(project_metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved metadata to {output_file}")

if __name__ == "__main__":
    # 获取当前脚本所在目录的上一级目录（即项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    cbench_dir = os.path.join(root_dir, "data/project/CBench")
    metadata_dir = os.path.join(root_dir, "data/c_metadata")
    process_projects(cbench_dir, metadata_dir)
