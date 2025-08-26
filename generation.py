import os
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import time # For exponential backoff/retry delay

# Add pandas for Excel handling
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not installed. Run: pip install pandas openpyxl")

# Updated LangChain imports to fix deprecation warning
try:
    from langchain_ollama import ChatOllama
    print("Using updated langchain-ollama package")
except ImportError:
    from langchain_community.chat_models import ChatOllama
    print("WARNING: Using deprecated ChatOllama. Run: pip install -U langchain-ollama")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict # For LangGraph state

# --- Configuration ---
@dataclass
class TestAutomationConfig:
    """Complete configuration for the universal test automation framework"""
    # Ollama Configuration
    ollama_model: str = "llama3:8b" # Using llama3:8b as per your available models
    ollama_base_url: str = "http://localhost:11434" # Default Ollama API URL
    temperature: float = 0.1 # Lower temperature for more consistent JSON output
    max_tokens: int = 4096
    max_llm_retries: int = 3 # Max retries for LLM to produce valid JSON
    llm_retry_delay_seconds: int = 5 # Delay between LLM retries

    # Input/Output Directories
    input_data_dir: str = "/Users/albertohernandez/Documents/projects/test-cases/data"
    output_dir: str = "ollama_generated_tests"

    # Excel Processing Configuration - More flexible column mapping
    excel_column_map: Optional[Dict[str, str]] = field(default_factory=lambda: {
        "test_case_id": "Requirement #",
        "title": "Test Scenario",
        "description": "Test Scenario",
        "steps": "Test Steps",
        "expected_result": "Expected Result",
        "qa_owner": "QA Owner",
        "test_data": "Test Data",
        "status": "Status",
        "url": "URL"  # Added for URL support
    })

    # Template for formatting extracted Excel data into natural language for the LLM
    nl_format_template: str = (
        "Test Case ID: {test_case_id}\n"
        "Test Case Title: {title}\n"
        "{description_section}"
        "{steps_section}"
        "{expected_result_section}"
        "{url_section}"
        "{other_info_section}"
        "\n---\n\n"
    )

    # Chunk size for processing large amounts of test cases
    test_case_chunk_size: int = 5  # Reduced chunk size for LLM processing
    max_successful_test_cases_to_output: int = 5 # Limit to 5 successful cases

    # Default URL for tests if not provided
    default_url: str = "https://example.com"


# Initialize configuration
config = TestAutomationConfig()

# Create comprehensive output directory structure
directories = [
    config.output_dir,
    f"{config.output_dir}/features",
    f"{config.output_dir}/tests",
    f"{config.output_dir}/input_copies"
]
for directory in directories:
    os.makedirs(directory, exist_ok=True)
print(f"Output directory structure created at {config.output_dir}")

# --- LangGraph State Definition ---
class TestAutomationState(TypedDict):
    """Complete state for the LangGraph multi-agent workflow"""
    original_nl_content: str 
    filename: str 
    analysis: Dict[str, Any] 
    generated_code: List[str] 
    errors: List[str] 
    current_step: str 
    test_cases_count: int  # Track number of test cases processed
    current_chunk: int     # Track current chunk being processed
    all_test_cases: List[Dict]  # Aggregate all test cases
    all_urls: List[str]  # Aggregate all URLs

# --- Agent Implementations ---

# Initialize Ollama LLM
try:
    ollama_llm = ChatOllama(model=config.ollama_model, base_url=config.ollama_base_url, temperature=config.temperature)
    llm_available = True
    print(f"Ollama LLM '{config.ollama_model}' initialized successfully.")
except Exception as e:
    ollama_llm = None
    llm_available = False
    print(f"ERROR: Failed to initialize Ollama LLM: {e}. Falling back to dummy responses for LLM-dependent agents.")


def input_agent(state: TestAutomationState) -> TestAutomationState:
    """Agent 1: Reads test cases from various file types with improved error handling."""
    print(f"--- Agent 1: Input Agent - Reading test case file: {os.path.basename(state['filename'])}")
    file_path = state["filename"]
    file_extension = Path(file_path).suffix.lower()
    
    nl_content = ""
    all_test_cases = []
    
    try:
        if file_extension in ('.xlsx', '.xls'):
            if not HAS_PANDAS:
                raise ImportError("pandas library required for Excel processing. Install with: pip install pandas openpyxl")
            
            xls = pd.ExcelFile(file_path)
            
            for sheet_name in xls.sheet_names:
                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    
                    # More aggressive cleaning
                    df = df.dropna(how='all').dropna(axis=1, how='all')
                    
                    if df.empty:
                        print(f"   - WARNING: Sheet '{sheet_name}' is empty after cleaning. Skipping.")
                        continue

                    print(f"   - Processing sheet '{sheet_name}' ({len(df)} rows)")
                    
                    # Use flexible column mapping
                    column_mapping = detect_test_columns_flexible(df, config.excel_column_map)
                    
                    if not column_mapping.get('steps') and not column_mapping.get('title'):
                        print(f"   - WARNING: No testable content found in sheet '{sheet_name}'. Skipping.")
                        continue
                    
                    test_cases_from_sheet = extract_test_cases_flexible(df, column_mapping)
                    all_test_cases.extend(test_cases_from_sheet)
                    print(f"   - INFO: Extracted {len(test_cases_from_sheet)} test cases from '{sheet_name}'")
                    
                except Exception as sheet_error:
                    print(f"   - ERROR: Processing sheet '{sheet_name}': {sheet_error}")
                    continue
            
            if not all_test_cases:
                raise ValueError(f"No valid test cases found across all sheets in Excel file.")

            nl_content = format_test_cases_as_natural_language(all_test_cases, config.nl_format_template)
            
            # --- CRUCIAL CHANGE: Redirect to processed .txt file ---
            # After creating the formatted NL content from Excel,
            # save it and then treat it as the new input for subsequent agents.
            copy_filename = os.path.basename(file_path) + ".processed.txt"
            copy_path = os.path.join(config.output_dir, "input_copies", copy_filename)
            with open(copy_path, 'w', encoding='utf-8') as f_copy:
                f_copy.write(nl_content)
            
            state["filename"] = copy_path # Now point to the processed text file
            file_extension = ".txt" # Treat it as a text file for parsing agent
            print(f"   -> Redirecting processing to generated text file: '{os.path.basename(copy_path)}'")


        elif file_extension in ('.txt', '.md', '.adoc', '.json', '.yaml', '.yml', '.csv'):
            with open(file_path, 'r', encoding='utf-8') as f:
                nl_content = f.read()
            print(f"   - Read text file: '{os.path.basename(file_path)}' ({len(nl_content)} characters)")
            # For these files, the original content IS the natural language content
            # No need to create a copy unless it's a first-time processed Excel
            if ".processed.txt" not in os.path.basename(file_path): # Avoid double saving if already processed
                copy_path = os.path.join(config.output_dir, "input_copies", os.path.basename(file_path))
                with open(copy_path, 'w', encoding='utf-8') as f_copy:
                    f_copy.write(nl_content)

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        state["original_nl_content"] = nl_content
        state["current_step"] = "input_read"
        state["test_cases_count"] = len(all_test_cases) if (file_extension in ('.xlsx', '.xls') or (file_extension == '.txt' and ".processed.txt" in state["filename"])) else 1
        state["current_chunk"] = 0
        state["all_test_cases"] = []  # Initialize
        state["all_urls"] = []  # Initialize
        
        print(f"--- Input Agent: Successfully processed '{os.path.basename(state['filename'])}'. Total test cases: {state['test_cases_count']}")
        return state
        
    except Exception as e:
        error_msg = f"Input Agent failed: {str(e)}"
        state["errors"].append(error_msg)
        state["current_step"] = "input_failed"
        print(f"--- ERROR: {error_msg}")
        return state


def detect_test_columns_flexible(df: pd.DataFrame, user_mapping: Dict[str, str]) -> Dict[str, str]:
    """More flexible column detection that handles messy spreadsheets."""
    columns_lower = {col: str(col).lower().strip() for col in df.columns}
    
    patterns = {
        'test_case_id': ['id', 'case', 'number', 'req', '#', 'tc', 'test id'],
        'title': ['title', 'name', 'scenario', 'description', 'summary', 'test case'],
        'steps': ['step', 'action', 'procedure', 'instruction', 'how', 'method'],
        'expected_result': ['expected', 'result', 'outcome', 'validation', 'verify'],
        'description': ['desc', 'detail', 'objective', 'purpose', 'summary'],
        'status': ['status', 'state', 'condition'],
        'priority': ['priority', 'importance', 'level'],
        'url': ['url', 'link', 'page', 'site']
    }
    
    mapping = {}
    
    # First try user-defined mapping
    if user_mapping:
        for standard_name, user_col in user_mapping.items():
            if user_col in df.columns:
                # Check if column has meaningful content
                if not df[user_col].dropna().empty:
                    mapping[standard_name] = user_col
    
    # Fill gaps with pattern matching
    for standard_name, keywords in patterns.items():
        if standard_name in mapping:
            continue
            
        best_match = None
        best_score = 0
        
        for actual_col, col_lower in columns_lower.items():
            if actual_col in mapping.values():
                continue
                
            # Score based on keyword matches
            score = sum(1 for keyword in keywords if keyword in col_lower)
            if score > best_score:
                # Additional check: column should have some non-empty content
                if not df[actual_col].dropna().empty:
                    best_score = score
                    best_match = actual_col
        
        if best_match:
            mapping[standard_name] = best_match
    
    # Ensure we have at least title or steps
    if not mapping.get('title') and not mapping.get('steps'):
        # Fallback to first text column with content
        for col in df.columns:
            if df[col].dtype == 'object' and not df[col].dropna().empty:
                if not mapping.get('title'):
                    mapping['title'] = col
                    break
    
    print(f"   - INFO: Column mapping: {mapping}")
    return mapping


def extract_test_cases_flexible(df: pd.DataFrame, column_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
    """More flexible test case extraction that handles incomplete data."""
    test_cases = []
    df_filled = df.fillna('')

    for idx, row in df_filled.iterrows():
        test_case = {}
        
        # Extract mapped columns
        for standard_name, actual_col in column_mapping.items():
            if actual_col in df.columns:
                value = str(row[actual_col]).strip()
                if value and value.lower() not in ['nan', 'none', '']:
                    test_case[standard_name] = value
        
        # Generate title if missing
        if not test_case.get('title'):
            if test_case.get('test_case_id'):
                test_case['title'] = f"Test Case {test_case['test_case_id']}"
            else:
                test_case['title'] = f"Test Case Row {idx + 1}"
        
        # For cases without explicit steps, use available information
        if not test_case.get('steps'):
            # Try to construct steps from other fields
            constructed_steps = []
            if test_case.get('description') and test_case['description'] != test_case.get('title'):
                constructed_steps.append(f"Execute: {test_case['description']}")
            if test_case.get('expected_result'):
                constructed_steps.append(f"Verify: {test_case['expected_result']}")
            
            if constructed_steps:
                test_case['steps'] = " | ".join(constructed_steps)
            elif test_case.get('title'):
                test_case['steps'] = f"Execute test: {test_case['title']}"
        
        # Only skip if we have absolutely no useful information
        if not any([test_case.get('title'), test_case.get('steps'), test_case.get('description')]):
            print(f"   - WARNING: Skipping empty row {idx + 1}")
            continue
            
        test_cases.append(test_case)
    
    return test_cases


def format_test_cases_as_natural_language(test_cases: List[Dict[str, Any]], template: str) -> str:
    """Format test cases with better handling of missing fields."""
    if not test_cases:
        return "No test cases to format."
    
    nl_content = ""
    
    for i, test_case in enumerate(test_cases):
        tc_id = test_case.get('test_case_id', f'TC_{i+1}')
        title = test_case.get('title', f"Test Case {i+1}")
        description = test_case.get('description', '')
        steps = test_case.get('steps', 'No specific steps provided')
        expected_result = test_case.get('expected_result', '')
        url = test_case.get('url', config.default_url)
        
        description_section = f"Description: {description}\n" if description and description != title else ""
        steps_section = f"Steps: {steps}\n"
        expected_result_section = f"Expected Result: {expected_result}\n" if expected_result else ""
        url_section = f"URL: {url}\n" if url else ""
        
        formatted_case = template.format(
            test_case_id=tc_id,
            title=title,
            description_section=description_section,
            steps_section=steps_section,
            expected_result_section=expected_result_section,
            url_section=url_section,
            other_info_section=""
        )
        
        nl_content += formatted_case
    
    return nl_content


def parsing_comprehension_agent(state: TestAutomationState) -> TestAutomationState:
    """Agent 2: Process test cases in chunks to handle large datasets with improved parsing."""
    print("--- Agent 2: Parsing & Comprehension Agent - Processing in chunks...")
    
    if not llm_available:
        print("WARNING: LLM not available. Using dummy parsing response.")
        # Generate dummy response proportional to input size
        num_cases = min(state.get("test_cases_count", 10), config.max_successful_test_cases_to_output)
        dummy_test_cases = []
        for i in range(num_cases):
            dummy_test_cases.append({
                "id": f"TC_{i+1:03d}",
                "title": f"Test Case {i+1}",
                "description": f"Automated test case {i+1} from {state['filename']}",
                "steps": [
                    {"action": "navigate", "target": config.default_url, "type": "page"},
                    {"action": "assert", "type": "title_contains", "expected_value": "Example"}
                ]
            })
        
        dummy_analysis = {
            "title": f"Test Suite for {state['filename']}",
            "description": f"Generated test suite with {len(dummy_test_cases)} test cases",
            "test_cases": dummy_test_cases,
            "urls_extracted": [config.default_url]
        }
        state["analysis"] = dummy_analysis
        state["all_test_cases"] = dummy_test_cases
        state["all_urls"] = [config.default_url]
        state["current_step"] = "parsed_dummy"
        print(f"--- Parsing Agent: Generated {len(dummy_test_cases)} dummy test cases")
        return state

    # Split the natural language content into manageable chunks
    nl_content = state["original_nl_content"]
    
    # Each "---" in the NL content represents a separate test case.
    test_cases_raw = nl_content.split("\n---\n")
    test_cases_raw = [tc.strip() for tc in test_cases_raw if tc.strip()]
    
    if not test_cases_raw:
        error_msg = "No parsable test cases found in the input content."
        state["errors"].append(error_msg)
        state["current_step"] = "parsing_failed"
        print(f"ERROR: {error_msg}")
        return state

    if len(test_cases_raw) <= config.test_case_chunk_size:
        # Small enough to process in one go
        return process_single_chunk(state, nl_content)
    else:
        # Need to chunk the content
        return process_in_chunks(state, test_cases_raw)


def process_single_chunk(state: TestAutomationState, nl_content: str) -> TestAutomationState:
    """Process content as a single chunk with improved parsing for observe steps."""
    print("   - Processing as single chunk")
    
    system_message_content = """You are an expert test automation engineer. Convert natural language test cases into structured JSON format.

CRITICAL: Respond ONLY with valid JSON. No explanations, no markdown, no extra text. Ensure all key-value pairs are separated by commas, and all property names and string values are enclosed in double quotes. Do not omit commas between objects or arrays.

Expected JSON structure:
{
  "title": "Overall test suite title",
  "description": "Brief description of what these tests cover",
  "test_cases": [
    {
      "id": "unique_test_id",
      "title": "test case title",
      "description": "what this test does",
      "steps": [
        {
          "action": "navigate|input|click|assert|wait",
          "target": "URL or CSS selector",
          "value": "input value (for input actions)",
          "type": "assertion type (for assert actions)",
          "expected_value": "expected value (for assertions)"
        }
      ],
      "expected_result": "expected outcome of the test"
    }
  ],
  "urls_extracted": ["list of URLs found in the text"]
}

Rules:
- If no URL is specified, use "https://example.com" as the default for navigation.
- For missing selectors, infer reasonable ones:
  - "pop-up notification" -> "div.notification" or "div[role='alert']"
  - "login button" -> "button[type='submit']" or "#login-btn"
  - "search box" -> "input[type='search']" or "#search"
  - "username field" -> "input[name='username']" or "#username"
- Convert "observe" or "monitor" steps to "wait" actions with a target selector (e.g., "wait for div.notification to be visible").
- If no assertion is provided, add a default assertion (e.g., check element visibility after a click).
- Ensure every test case has at least one actionable step.
- For expected results like "redirect to the loan", infer a URL like "/loan" and add an assert step for url_contains.
"""

    response = "" # Initialize response for each attempt
    for attempt in range(config.max_llm_retries):
        try:
            messages = [
                SystemMessage(content=system_message_content),
                HumanMessage(content=f"Convert these test cases to JSON:\n\n{nl_content}")
            ]
            
            if attempt > 0:
                print(f"   - Retrying LLM call for single chunk (Attempt {attempt + 1}/{config.max_llm_retries})...")
                messages.append(HumanMessage(content=f"Your previous response was not valid JSON. Please provide ONLY the JSON object, without any additional text. Here was your previous (invalid) response:\n\n{response}"))
            
            chain = ChatPromptTemplate.from_messages(messages) | ollama_llm | StrOutputParser()
            response = chain.invoke({})
            
            # Preliminary check for JSON-like structure
            if not response.strip().startswith('{') or not response.strip().endswith('}'):
                raise ValueError("Response does not resemble a JSON object")
            
            structured_analysis = extract_and_parse_json(response)
            
            # Enhance test cases with defaults for observe and other vague steps
            for test_case in structured_analysis.get('test_cases', []):
                if not any(step.get('action') == 'navigate' for step in test_case.get('steps', [])):
                    test_case['steps'].insert(0, {
                        "action": "navigate",
                        "target": config.default_url,
                        "type": "page"
                    })
                for step in test_case.get('steps', []):
                    if step.get('action') == 'manual' or 'observe' in step.get('action', '').lower() or 'monitor' in step.get('action', '').lower():
                        step['action'] = 'wait'
                        step['target'] = "div.notification" if "notification" in test_case.get('description', '').lower() else step.get('target', 'body')
                        step['value'] = ""
                        step['type'] = "visible"
                        step['expected_value'] = "true"
                    if step.get('action') == 'click' and not step.get('target'):
                        step['target'] = "div.notification" if "notification" in test_case.get('description', '').lower() else "button"
                        step['type'] = "element_visible"
                        step['expected_value'] = "false"  # Assume click hides
            state["analysis"] = structured_analysis
            state["current_step"] = "parsed_comprehended"
            test_count = len(structured_analysis.get('test_cases', []))
            print(f"--- Parsing Agent: Extracted {test_count} test cases (Attempt {attempt + 1})")
            return state
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"   - ERROR: JSON parsing failed (Attempt {attempt + 1}): {str(e)}")
            if attempt < config.max_llm_retries - 1:
                time.sleep(config.llm_retry_delay_seconds)
                continue
            else:
                error_msg = f"Single chunk parsing failed after {config.max_llm_retries} retries: {str(e)}\nRaw LLM response (first 2000 chars): {response[:2000]}..."
                state["errors"].append(error_msg)
                state["current_step"] = "parsing_failed"
                return state

    return state


def fallback_parse_nl_content(state: TestAutomationState, nl_content: str) -> TestAutomationState:
    """Fallback parsing when LLM JSON output fails."""
    print("   - INFO: Attempting fallback parsing of natural language content")
    test_cases = []
    test_cases_raw = nl_content.split("\n---\n")
    for i, tc_raw in enumerate(test_cases_raw):
        if not tc_raw.strip():
            continue
        test_case = {}
        lines = tc_raw.split("\n")
        for line in lines:
            if line.startswith("Test Case ID: "):
                test_case["id"] = line.split(": ", 1)[1]
            elif line.startswith("Test Case Title: "):
                test_case["title"] = line.split(": ", 1)[1]
            elif line.startswith("Steps: "):
                steps_str = line.split(": ", 1)[1]
                if "observe" in steps_str.lower() or "monitor" in steps_str.lower():
                    test_case["steps"] = [{"action": "wait", "target": "div.notification", "type": "visible", "expected_value": "true"}]
                else:
                    test_case["steps"] = [{"action": "manual", "target": "unknown", "value": ""}]
            elif line.startswith("Expected Result: "):
                test_case["expected_result"] = line.split(": ", 1)[1]
        test_cases.append(test_case)
    
    state["analysis"] = {
        "title": f"Fallback Test Suite for {state['filename']}",
        "description": "Generated from fallback parsing due to LLM JSON failure",
        "test_cases": test_cases[:config.max_successful_test_cases_to_output],
        "urls_extracted": []
    }
    state["current_step"] = "parsed_fallback"
    print(f"--- Parsing Agent: Extracted {len(test_cases)} test cases via fallback")
    return state


def process_in_chunks(state: TestAutomationState, test_cases_raw: List[str]) -> TestAutomationState:
    """Process large content in chunks and generate code for each chunk."""
    print("   - Processing in chunks due to large content size")
    
    # Group into chunks based on test_case_chunk_size
    chunks_of_raw_test_cases = [
        test_cases_raw[i:i + config.test_case_chunk_size]
        for i in range(0, len(test_cases_raw), config.test_case_chunk_size)
    ]
    
    print(f"   - Split into {len(chunks_of_raw_test_cases)} chunks, each with max {config.test_case_chunk_size} test cases")
    
    for i, chunk_raw_test_cases in enumerate(chunks_of_raw_test_cases):
        print(f"   - Processing chunk {i+1}/{len(chunks_of_raw_test_cases)}")
        chunk_nl_content = "\n---\n".join(chunk_raw_test_cases)
        
        # Create a temporary state for processing this chunk
        temp_state_for_chunk = {
            "original_nl_content": chunk_nl_content,
            "filename": state["filename"],
            "analysis": {},
            "generated_code": [],
            "errors": [],
            "current_step": "initial",
            "test_cases_count": len(chunk_raw_test_cases),
            "current_chunk": i + 1,
            "all_test_cases": [],
            "all_urls": []
        }
        
        try:
            # Process the chunk
            chunk_result_state = process_single_chunk(temp_state_for_chunk, chunk_nl_content)
            
            if chunk_result_state["current_step"] == "parsing_failed":
                print(f"   - WARNING: Chunk {i+1} failed, adding errors and continuing with other chunks.")
                state["errors"].extend(chunk_result_state["errors"])
                continue
            
            chunk_analysis = chunk_result_state["analysis"]
            chunk_test_cases = chunk_analysis.get("test_cases", [])
            chunk_urls = chunk_analysis.get("urls_extracted", [])
            
            # Ensure unique IDs across chunks
            for j, test_case in enumerate(chunk_test_cases):
                if not test_case.get("id"):
                    test_case["id"] = f"CHUNK{i+1}_TC_{j+1:03d}"
            
            # Aggregate test cases and URLs
            state["all_test_cases"].extend(chunk_test_cases)
            state["all_urls"].extend(chunk_urls)
            
            print(f"   - INFO: Chunk {i+1}: Extracted {len(chunk_test_cases)} test cases")
            
            # Update analysis with chunk-specific data for code generation
            chunk_result_state["analysis"] = {
                "title": f"Chunk {i+1} Test Suite for {state['filename']}",
                "description": f"Tests from chunk {i+1}/{len(chunks_of_raw_test_cases)}",
                "test_cases": chunk_test_cases,
                "urls_extracted": chunk_urls
            }
            
            # Generate and write test files for this chunk
            chunk_result_state = code_generation_agent(chunk_result_state)
            
            # Update state with generated code and any new errors
            state["generated_code"].extend(chunk_result_state["generated_code"])
            state["errors"].extend(chunk_result_state["errors"])
            
        except Exception as e:
            error_msg = f"Chunk processing failed for chunk {i+1}: {str(e)}"
            state["errors"].append(error_msg)
            print(f"   - ERROR: {error_msg}")
    
    # Combine all results
    final_analysis = {
        "title": f"Test Suite for {state['filename']} ({len(state['all_test_cases'])} total test cases)",
        "description": f"Comprehensive test suite processed in {len(chunks_of_raw_test_cases)} chunks.",
        "test_cases": state['all_test_cases'],
        "urls_extracted": list(set(state['all_urls']))  # Remove duplicates
    }
    
    state["analysis"] = final_analysis
    state["current_step"] = "parsed_comprehended"
    state["test_cases_count"] = len(state['all_test_cases'])
    print(f"--- Parsing Agent: Total extracted {len(state['all_test_cases'])} test cases from {len(chunks_of_raw_test_cases)} chunks")
    return state


def extract_and_parse_json(response: str) -> dict:
    """Enhanced JSON extraction with better error handling."""
    cleaned_response = response.strip()
    
    # Remove common LLM artifacts that wrap JSON
    patterns_to_remove = [
        r'^\s*```(?:json)?\s*\n',  # Leading ```json
        r'\n\s*```\s*$',         # Trailing ``` with optional newline
        r'^Here.*?JSON.*?:\s*',  # "Here is the JSON:"
        r'^The.*?JSON.*?:\s*',   # "The JSON output is:"
        r'^Response:\s*',        # "Response:"
        r'^\s*[\w\s]*?```json\s*\n', # Catches "Here's the JSON ```json
        r'\n```\s*$',                # Catches trailing ```
    ]
    
    for pattern in patterns_to_remove:
        cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL).strip()
    
    # Try to find JSON object boundaries (first '{' to last '}')
    json_match = re.search(r'(\{.*\})', cleaned_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = cleaned_response
    
    # Try parsing raw JSON first
    try:
        structured_data = json.loads(json_str)
        print("   - INFO: Raw JSON parsed successfully without fixes")
        # Validate structure
        if not isinstance(structured_data, dict):
            raise ValueError("LLM response is not a valid JSON object (expected dict).")
        if 'test_cases' not in structured_data or not isinstance(structured_data['test_cases'], list):
            structured_data['test_cases'] = []
            print("   - WARNING: 'test_cases' key missing or not a list in LLM's JSON output. Defaulting to empty list.")
        if 'urls_extracted' not in structured_data or not isinstance(structured_data['urls_extracted'], list):
            structured_data['urls_extracted'] = []
            print("   - WARNING: 'urls_extracted' key missing or not a list in LLM's JSON output. Defaulting to empty list.")
        return structured_data
    except json.JSONDecodeError as raw_e:
        print(f"   - INFO: Raw JSON parsing failed: {str(raw_e)}. Attempting to fix JSON...")
    
    # Apply fixes if raw parsing fails
    json_str = fix_common_json_issues(json_str)
    
    # Parse JSON
    try:
        structured_data = json.loads(json_str)
        # Validate structure
        if not isinstance(structured_data, dict):
            raise ValueError("LLM response is not a valid JSON object (expected dict).")
        if 'test_cases' not in structured_data or not isinstance(structured_data['test_cases'], list):
            structured_data['test_cases'] = []
            print("   - WARNING: 'test_cases' key missing or not a list in LLM's JSON output. Defaulting to empty list.")
        if 'urls_extracted' not in structured_data or not isinstance(structured_data['urls_extracted'], list):
            structured_data['urls_extracted'] = []
            print("   - WARNING: 'urls_extracted' key missing or not a list in LLM's JSON output. Defaulting to empty list.")
        return structured_data
    except json.JSONDecodeError as e:
        print(f"   - ERROR: JSON parsing failed: {str(e)}")
        print(f"   - INFO: Raw LLM response (full): {response}")
        print(f"   - INFO: Cleaned JSON (full): {json_str}")
        raise e
    except ValueError as e:
        print(f"   - ERROR: Value Error in JSON parsing: {str(e)}")
        print(f"   - INFO: Raw LLM response (full): {response}")
        print(f"   - INFO: Cleaned JSON (full): {json_str}")
        raise e


def fix_common_json_issues(json_str: str) -> str:
    """Fix common JSON formatting issues from LLM responses carefully."""
    # 1. Remove comments (// or /* ... */)
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)
    json_str = re.sub(r"//.*", "", json_str)

    # 2. Replace single quotes with double quotes for string values
    json_str = json_str.replace("'", '"')

    # 3. Quote unquoted property names (e.g., key: → "key":)
    json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)

    # 4. Fix missing commas between objects in arrays (e.g., } { → }, {)
    json_str = re.sub(r'}\s*{', '}, {', json_str)

    # 5. Fix missing commas between key-value pairs (e.g., "value""key" → "value","key")
    json_str = re.sub(r'("[^"]*")(\s*"[a-zA-Z_][a-zA-Z0-9_]*":)', r'\1,\2', json_str)

    # 6. Remove trailing commas before closing brackets/braces
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)

    # 7. Ensure proper array/object closure before next key
    json_str = re.sub(r'([\]\}])\s*("[a-zA-Z_][a-zA-Z0-9_]*":)', r'\1,\2', json_str)

    # 8. Final cleanup: Remove any leading/trailing text after the last closing brace
    last_brace_index = json_str.rfind('}')
    if last_brace_index != -1:
        json_str = json_str[:last_brace_index + 1]

    # 9. Debugging: Log the cleaned JSON
    print(f"   - DEBUG: Cleaned JSON (first 2000 chars): {json_str[:2000]}...")

    return json_str


def code_generation_agent(state: TestAutomationState) -> TestAutomationState:
    """Agent 4: Generate multiple Playwright test files."""
    print("--- Agent 4: Code Generation Agent - Generating Playwright Python code...")
    
    analysis = state["analysis"]
    test_cases = analysis.get('test_cases', [])
    
    if not test_cases:
        print("   - WARNING: No test cases to generate code for")
        state["generated_code"].append("# No test cases found")
        state["current_step"] = "code_generated"
        return state

    test_output_dir = os.path.join(config.output_dir, "tests")
    os.makedirs(test_output_dir, exist_ok=True) # Ensure dir exists
    chunk_generated_code = []
    
    # Limit test cases for output
    test_cases_to_output = test_cases[:config.max_successful_test_cases_to_output]

    # Generate individual test files for each test case
    for i, test_case in enumerate(test_cases_to_output):
        test_id = test_case.get('id', f'TC_{i+1:03d}')
        title = test_case.get('title', f'Test Case {i+1}')
        description = test_case.get('description', '')
        steps = test_case.get('steps', [])
        
        # Create safe filename
        safe_filename = re.sub(r'[^a-zA-Z0-9_]', '_', test_id.lower())
        test_file_name = f"test_{safe_filename}.py"
        
        # Generate test code
        test_code = generate_playwright_test(test_id, title, description, steps, Path(state['filename']).name)
        
        # Save individual test file
        test_file_path = os.path.join(test_output_dir, test_file_name)
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        chunk_generated_code.append(test_code)
        print(f"   - INFO: Generated: {test_file_name}")

    state["generated_code"].extend(chunk_generated_code)
    state["current_step"] = "code_generated"
    print(f"--- Code Generation Agent: Created {len(test_cases_to_output)} test files for chunk {state['current_chunk']}")
    return state


def generate_playwright_test(test_id: str, title: str, description: str, steps: List[Dict], source_file: str) -> str:
    """Generate individual Playwright test code with improved handling for observe steps."""
    safe_function_name = re.sub(r'[^a-zA-Z0-9_]', '_', test_id.lower())
    
    code = f'''import pytest
from playwright.sync_api import Page, expect
import time
import re # Import re for regex assertions

@pytest.fixture(scope="function") 
def browser_page(page: Page):
    """Provides a Playwright page object to tests."""
    # Set reasonable timeouts
    page.set_default_timeout(30000)  # 30 seconds
    page.set_default_navigation_timeout(60000)  # 60 seconds
    yield page

def test_{safe_function_name}(browser_page: Page):
    """
    Test ID: {test_id}
    Title: {title}
    Description: {description}
    Source: {source_file}
    """
'''

    if not steps:
        code += '''    # No specific steps provided
    pytest.skip("No automation steps defined for this test case")
'''
        return code

    # Generate step implementations
    for step in steps:
        action = step.get('action', '').lower()
        target = step.get('target', '')
        value = step.get('value', '')
        expected_value = step.get('expected_value', '')
        step_type = step.get('type', '')

        code += '\n    # ' + str(step) + '\n'

        if action == 'navigate' and target:
            code += f'    browser_page.goto("{target}")\n'
            code += '    browser_page.wait_for_load_state("networkidle")\n'
            
        elif action == 'input' and target and value:
            code += f'    browser_page.locator("{target}").fill("{value}")\n'
            
        elif action == 'click' and target:
            code += f'    browser_page.locator("{target}").click()\n'
            code += '    time.sleep(1)  # Allow for UI updates\n'
            
        elif action == 'wait' and target:
            code += f'    browser_page.locator("{target}").wait_for(state="visible", timeout=10000)\n'
            
        elif action == 'assert':
            if step_type == 'title_equals' and expected_value:
                code += f'    expect(browser_page).to_have_title("{expected_value}")\n'
            elif step_type == 'title_contains' and expected_value:
                code += f'    expect(browser_page).to_have_title(re.compile(r".*{re.escape(expected_value)}.*"))\n'
            elif step_type == 'url_equals' and expected_value:
                code += f'    expect(browser_page).to_have_url("{expected_value}")\n'
            elif step_type == 'url_contains' and expected_value:
                code += f'    expect(browser_page.url).to_contain("{expected_value}")\n'
            elif step_type == 'element_visible' and target:
                code += f'    expect(browser_page.locator("{target}")).to_be_visible()\n'
            elif step_type == 'text_contains' and target and expected_value:
                code += f'    expect(browser_page.locator("{target}")).to_contain_text("{expected_value}")\n'
            elif step_type == 'attribute_equals' and target and expected_value:
                attribute_name = step.get('attribute_name', 'value') # Default to 'value'
                code += f'    expect(browser_page.locator("{target}")).to_have_attribute("{attribute_name}", "{expected_value}")\n'
            else:
                code += f'    # Assertion: {step_type} - {target} - {expected_value}\n'
                code += '    # TODO: Implement custom assertion logic\n'
                code += '    pass\n'
        else:
            code += f'    # Unhandled step: {action}\n'
            code += '    # TODO: Implement custom step logic\n'
            code += '    pass\n'

    return code


# --- LangGraph Workflow Definition ---
def build_test_automation_workflow() -> StateGraph:
    """Build the complete LangGraph workflow with all 5 agents."""
    workflow = StateGraph(TestAutomationState)

    # Add agent nodes
    workflow.add_node("input", input_agent)
    workflow.add_node("parsing_comprehension", parsing_comprehension_agent)
    workflow.add_node("code_generation", code_generation_agent)

    # Set entry point
    workflow.set_entry_point("input")

    # Define sequential edges
    workflow.add_edge("input", "parsing_comprehension")
    workflow.add_edge("parsing_comprehension", "code_generation")
    workflow.add_edge("code_generation", END)

    return workflow

# Compile the workflow
compiled_workflow = build_test_automation_workflow().compile()
print("LangGraph workflow built and compiled successfully!")

# --- Main Execution Loop ---
def main():
    print(f"\n--- Starting Automated Test Generation with Ollama & LangGraph ---")
    print(f"Input data directory: {config.input_data_dir}")
    print(f"Output directory: {config.output_dir}")
    print(f"Ollama Model: {config.ollama_model} at {config.ollama_base_url}")
    print("-----------------------------------------------------------------\n")

    input_files = []
    # Find all supported files in the input directory
    for root, _, files in os.walk(config.input_data_dir):
        for file in files:
            file_extension = Path(file).suffix.lower()
            if file_extension in ('.xlsx', '.xls', '.txt', '.md', '.adoc', '.json', '.yaml', '.yml', '.csv'):
                input_files.append(os.path.join(root, file))

    if not input_files:
        print(f"No supported test case files found in '{config.input_data_dir}'.")
        print("Please add some files (e.g., .xlsx, .txt, .md) with test descriptions.")
        return

    results = []
    total_test_cases_processed = 0
    
    for i, file_path in enumerate(input_files):
        print(f"\n--- Processing file {i+1}/{len(input_files)}: {os.path.basename(file_path)} ---")
        
        initial_state: TestAutomationState = {
            "original_nl_content": "",
            "filename": file_path,
            "analysis": {},
            "generated_code": [],
            "errors": [],
            "current_step": "initial",
            "test_cases_count": 0,
            "current_chunk": 0,
            "all_test_cases": [],
            "all_urls": []
        }
        
        try:
            final_state = compiled_workflow.invoke(initial_state)
            results.append(final_state)
            total_test_cases_processed += final_state.get("test_cases_count", 0)
            
            print(f"\n--- Summary for {os.path.basename(final_state['filename'])} ---")
            print(f"Final Status: {final_state['current_step']}")
            print(f"Test Cases Processed by LLM: {final_state.get('test_cases_count', 0)}")
            print(f"Test Cases Generated: {len(final_state.get('generated_code', []))}")
            print(f"Errors: {len(final_state['errors'])}")
            
            if final_state['errors']:
                for error in final_state['errors']:
                    print(f"  - ERROR: {error}")
            
        except Exception as e:
            print(f"Workflow execution failed for {os.path.basename(file_path)}: {e}")
            initial_state["errors"].append(f"Workflow crashed: {str(e)}")
            initial_state["current_step"] = "crashed"
            results.append(initial_state)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    successful_runs = [r for r in results if r['current_step'] == 'code_generated' and not r['errors']]
    files_with_errors = [r for r in results if r['errors']]
    
    print(f"Files processed: {len(results)}/{len(input_files)}")
    print(f"Successfully completed: {len(successful_runs)}")
    print(f"Files with errors: {len(files_with_errors)}")
    print(f"Total test cases processed: {total_test_cases_processed}")
    
    if files_with_errors:
        print(f"\nFiles with errors:")
        for result in files_with_errors:
            print(f"  - {result['filename']}: {len(result['errors'])} errors")
    
    print(f"\nGenerated artifacts saved in: {config.output_dir}")
    print("  - Test code: tests/")
    print("  - Input copies: input_copies/")


if __name__ == "__main__":
    main()