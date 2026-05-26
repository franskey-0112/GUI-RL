import ast
import json
import re
from numbers import Real

SUPPORTED_ACTIONS = {
    "click",
    "type",
    "scroll",
    "complete",
    "open_app",
    "wait",
    "long_press",
    "moveto",
    "doubleclick",
    "impossible",
    "rightclick",
    "press",
}
ACTION_ALIASES = {
    "double_click": "doubleclick",
    "double-click": "doubleclick",
    "right_click": "rightclick",
    "right-click": "rightclick",
    "long_click": "long_press",
    "long-click": "long_press",
    "move_to": "moveto",
    "move-to": "moveto",
}
POINT_ACTIONS = {"click", "long_press", "moveto", "doubleclick", "rightclick"}
TEXT_MATCH_ACTIONS = {"type", "open_app"}
NO_INPUT_TEXT = {"", "no input text", "none", "null"}
SCROLL_DIRECTIONS = {"left", "right", "up", "down"}
PRESS_KEYS = {"back", "home", "enter", "space", "tab", "down", "page_down", "hotkey"}
PRESS_ACTION_TO_KEY = {
    "press_back": "BACK",
    "press_home": "HOME",
    "enter": "ENTER",
    "press_enter": "ENTER",
    "press_space": "SPACE",
    "press_tab": "TAB",
    "press_down": "DOWN",
    "press_pgdn": "PAGE_DOWN",
    "hotkey": "HOTKEY",
}
PRESS_KEY_ALIASES = {
    "": "",
    "no_input_text": "",
    "none": "",
    "null": "",
    "back": "BACK",
    "home": "HOME",
    "enter": "ENTER",
    "return": "ENTER",
    "space": "SPACE",
    "spacebar": "SPACE",
    "tab": "TAB",
    "down": "DOWN",
    "arrow_down": "DOWN",
    "pagedown": "PAGE_DOWN",
    "page_down": "PAGE_DOWN",
    "pgdn": "PAGE_DOWN",
    "hotkey": "HOTKEY",
}


def normalize_action(action):
    if action is None:
        return "no action"
    action = str(action).strip().lower()
    return ACTION_ALIASES.get(action, action)


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    return "" if text.lower() in NO_INPUT_TEXT else text


def normalize_press_key(text):
    text = "" if text is None else str(text).strip().lower()
    text = re.sub(r"[\s-]+", "_", text)
    return PRESS_KEY_ALIASES.get(text, text.upper())


def normalize_action_and_input(action, input_text):
    action = normalize_action(action)
    if action in PRESS_ACTION_TO_KEY:
        return "press", PRESS_ACTION_TO_KEY[action]
    if action == "press":
        return "press", normalize_press_key(input_text)
    if action == "scroll":
        return "scroll", normalize_text(input_text).upper()
    return action, normalize_text(input_text)


def _is_number(value):
    return isinstance(value, Real) and not isinstance(value, bool)


def _extract_answer_content(content):
    answer_tag_pattern = r"<answer>(.*?)</answer>"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        return content_answer_match.group(1).strip()
    return None


def _strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_answer_actions(content):
    answer_content = _extract_answer_content(content)
    if answer_content is None:
        return None

    answer_content = _strip_code_fence(answer_content)
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(answer_content)
            return [parsed] if isinstance(parsed, dict) else parsed
        except Exception:
            pass
    return None


def _coerce_point(point):
    if isinstance(point, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", point)
        if len(nums) >= 2:
            return [float(nums[0]), float(nums[1])]
        return None

    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    if not (_is_number(point[0]) and _is_number(point[1])):
        return None
    return [float(point[0]), float(point[1])]


def _is_sentinel_point(point):
    return point is not None and len(point) >= 2 and point[0] == -100 and point[1] == -100


def _first_action_dict(content):
    actions = parse_answer_actions(content)
    if not isinstance(actions, list) or len(actions) == 0:
        return None
    first = actions[0]
    return first if isinstance(first, dict) else None

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str = normalize_text(predicted_str).replace("[", "").replace("]", "")
    ground_truth_str = normalize_text(ground_truth_str).replace("[", "").replace("]", "")
    if not predicted_str and not ground_truth_str:
        return 1
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def extract_action(content):
    action = _first_action_dict(content)
    if action is not None and "action" in action:
        return normalize_action(action["action"])

    answer_content = _extract_answer_content(content)
    if answer_content is not None:
        action_match = re.search(r"""["']action["']\s*:\s*["']([^"']+)["']""", answer_content)
        if action_match:
            return normalize_action(action_match.group(1))
    return "no action"

def extract_input_text(content):
    action = _first_action_dict(content)
    if action is not None and "input_text" in action:
        return "" if action["input_text"] is None else str(action["input_text"])

    answer_content = _extract_answer_content(content)
    if answer_content is not None:
        action_match = re.search(r"""["']input_text["']\s*:\s*["'](.*?)["']""", answer_content, re.DOTALL)
        if action_match:
            return action_match.group(1)
    return "no input text"

def extract_coord(content):
    action = _first_action_dict(content)
    if action is not None:
        for key in ("point", "coord", "coordinate", "bbox"):
            if key in action:
                point = _coerce_point(action[key])
                if point is not None:
                    return point, True

    answer_content = _extract_answer_content(content)
    try:
        if answer_content:
            coord_match = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", answer_content)
            if coord_match:
                coord = [float(coord_match.group(1)), float(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r"\{.*\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*.*\}"
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [float(coord_match.group(1)), float(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False
    
def r1gui_format_reward(predict_str: str) -> float:
    """
    检查 predict_str 是否符合 <think></think><answer></answer> 的格式，
    并验证 <answer> 中的内容是否符合 [{'action': 'action', 'point': '[x,y]', 'input_text': 'no input text'}] 的格式要求。
    """
    predict_str = predict_str.strip()
    # 检查 <think> 和 <answer> 的外部结构
    outer_pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if not re.fullmatch(outer_pattern, predict_str):
        return 0.0

    actions = parse_answer_actions(predict_str)
    if not isinstance(actions, list) or len(actions) != 1:
        return 0.0

    action = actions[0]
    if not isinstance(action, dict):
        return 0.0
    if "action" not in action or "point" not in action or "input_text" not in action:
        return 0.0
    if not isinstance(action["action"], str):
        return 0.0
    pred_action, pred_input_text = normalize_action_and_input(action["action"], action["input_text"])
    if pred_action not in SUPPORTED_ACTIONS:
        return 0.0
    point = _coerce_point(action["point"])
    if point is None:
        return 0.0
    if not isinstance(action["input_text"], str):
        return 0.0
    if pred_action not in POINT_ACTIONS and not _is_sentinel_point(point):
        return 0.0
    if pred_action in {"complete", "wait", "impossible"} and normalize_text(pred_input_text) != "":
        return 0.0
    if pred_action == "open_app" and pred_input_text == "":
        return 0.0
    if pred_action == "scroll" and pred_input_text.lower() not in SCROLL_DIRECTIONS:
        return 0.0
    if pred_action == "press" and pred_input_text.lower() not in PRESS_KEYS:
        return 0.0

    return 1.0

def r1gui_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    比较 predict_str 和 ground_truth 中的动作和参数是否一致。
    """
    try:
        # 提取 ground_truth 的动作和参数
        ground_truth=json.loads(ground_truth)
        gt_action, gt_input_text = normalize_action_and_input(ground_truth['action'], ground_truth['input_text'])
        gt_bbox=ground_truth['gt_bbox']
        pred_action, pred_input_text = normalize_action_and_input(extract_action(predict_str), extract_input_text(predict_str))
        pred_bbox, has_coord=extract_coord(predict_str)
        
        if pred_action!=gt_action:
            return 0.0
        
        if gt_action in POINT_ACTIONS:
            if not has_coord:
                return 0.0
            if len(gt_bbox)==2:
                if (pred_bbox[0]-gt_bbox[0])**2+(pred_bbox[1]-gt_bbox[1])**2<70**2:
                    return 1.0
                else:
                    return 0.0
            elif len(gt_bbox)==4:
                if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0
        elif gt_action == "scroll":
            if not _is_sentinel_point(pred_bbox):
                return 0.0
            if pred_input_text == gt_input_text:
                return 1.0
            else:
                return 0.0
        elif gt_action == "press":
            if not _is_sentinel_point(pred_bbox):
                return 0.0
            if normalize_press_key(pred_input_text) == normalize_press_key(gt_input_text):
                return 1.0
            else:
                return 0.0
        elif gt_action in TEXT_MATCH_ACTIONS or normalize_text(gt_input_text):
            if gt_action in TEXT_MATCH_ACTIONS and not _is_sentinel_point(pred_bbox):
                return 0.0
            if calculate_f1_score(pred_input_text, gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        else:
            if not _is_sentinel_point(pred_bbox):
                return 0.0
            return 1.0 if normalize_text(pred_input_text) == "" else 0.0

    except Exception as e:
        return 0.0
    
def r1gui_compute_score(predict_str: str, ground_truth: str):
    format = r1gui_format_reward(predict_str)
    accuracy = r1gui_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.8 * accuracy + 0.2 * format,
        "format": format,
        "accuracy": accuracy,
    }


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    result = r1gui_compute_score(solution_str, ground_truth)
    return {
        "score": result["overall"],
        "overall": result["overall"],
        "format": result["format"],
        "accuracy": result["accuracy"],
    }

# pr=("<think> The command 'What's on the menu at IHOP?' suggests a search for information about the menu at an IHOP restaurant. However, "
# "the current UI screenshot is a calendar application displaying holidays and significant dates for the month of October and November. There is no direct way to per"
# "form a web search or access an IHOP menu from this calendar app. Therefore, the appropriate action would be to exit the current application and open a web browser"
# "or a dedicated app for searching the IHOP menu. "                                                                                                               
# "Since the action history is 'None', the first step is to navigate away from the current app to a web browser or a search engine.</think> "
# " <answer>[{'action': 'scroll', 'point': [123, 401], 'input_text': 'left'}]</answer>")
# gt=json.dumps({"action": "scroll", "gt_bbox": [103.0, 409.18800000000005], "input_text": "LEFT"})
# print(gr_iou_accuracy_reward(pr,gt))
# print(gr_format_reward(pr))
