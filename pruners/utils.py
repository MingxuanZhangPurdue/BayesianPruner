import re
from typing import Optional, List, Union

def _compile_target_modules(target_modules: Optional[Union[List[str], str]]) -> Optional[List[re.Pattern]]:
    if target_modules is None:
        return None
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    
    compiled_patterns = []
    for pattern in target_modules:
        if pattern.startswith("re:"):
            # If the pattern starts with 're:', compile it as a regular expression
            compiled_patterns.append(re.compile(pattern[3:], re.IGNORECASE))
        else:
            # Compile it to match the pattern anywhere in the string
            escaped_pattern = re.escape(pattern).replace(r'\*', '.*')
            compiled_patterns.append(re.compile(f".*{escaped_pattern}.*", re.IGNORECASE))
    
    return compiled_patterns