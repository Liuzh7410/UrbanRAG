"""
KG-Enhanced Geocoding with DCABG Algorithm
完整工作流实现

工作流程：
1. 地址解析和索引构建
2. 精确地址检索
3. 空间约束算法前置（确定Seed）
4. DCABG算法
5. LLM推理
"""

import os
import re
import csv
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from shapely.geometry import Point, Polygon, LineString, MultiLineString
from shapely.ops import unary_union, polygonize, transform, nearest_points
from shapely import wkt
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

# 加载环境变量
load_dotenv()

# ============================================================================
# 配置参数 - 可根据需要调整
# ============================================================================

CONFIG = {
    # 输入模式: "test" 或 "file"
    "input_mode": "file",  # 改为 "file" 以从CSV读取地址

    # 文件模式配置（当 input_mode = "file" 时使用）
    "input_file": "data_sample/test_random/tokyo_1000.csv",  # CSV文件路径
    "address_column": "place_name",                    # 地址所在列名
    "limit": None,  # 随机样本文件已固定，不再额外截断

    # 测试模式配置（当 input_mode = "test" 时使用）
    "test_addresses": [],

    # 输出配置
    "output_file": "results/geocoding_results_tokyo_1000.csv",

    # 显示配置
    "verbose": True,  # 是否显示详细处理过程
    "show_results": True  # 是否在终端显示每个结果
}

# 初始化Neo4j连接
neo4j_driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

# 初始化OpenAI客户端
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ============================================================================
# 坐标系统映射
# ============================================================================

def get_epsg_for_prefecture(pref: str) -> int:
    """
    根据都道府县返回对应的JGD2011平面直角坐标系EPSG代码

    日本被划分为19个平面直角坐标系统区域（Zone I-XIX）
    """
    # 都道府県 → EPSG代码映射
    prefecture_epsg_map = {
        # Zone I (EPSG:6669) - 北海道北部
        '北海道': 6670,  # 大部分使用Zone II

        # Zone III (EPSG:6671) - 东北北部
        '青森県': 6671,
        '秋田県': 6671,

        # Zone IV (EPSG:6672) - 东北南部
        '岩手県': 6672,
        '宮城県': 6672,
        '山形県': 6672,

        # Zone V (EPSG:6673) - 新潟
        '福島県': 6673,
        '新潟県': 6673,

        # Zone VI (EPSG:6674) - 北陆
        '石川県': 6674,
        '富山県': 6674,
        '福井県': 6674,

        # Zone VII (EPSG:6675) - 中部
        '長野県': 6675,
        '岐阜県': 6675,

        # Zone VIII (EPSG:6676) - 中部南部（静冈等）
        '静岡県': 6676,
        '山梨県': 6676,

        # Zone IX (EPSG:6677) - 关东（东京等）
        '東京都': 6677,
        '神奈川県': 6677,
        '千葉県': 6677,
        '埼玉県': 6677,
        '茨城県': 6677,
        '栃木県': 6677,
        '群馬県': 6677,

        # Zone X (EPSG:6678) - 近畿
        '愛知県': 6678,
        '三重県': 6678,

        # Zone XI (EPSG:6679) - 近畿中部
        '大阪府': 6679,
        '京都府': 6679,
        '兵庫県': 6679,
        '滋賀県': 6679,
        '奈良県': 6679,
        '和歌山県': 6679,

        # Zone XII (EPSG:6680) - 中国
        '鳥取県': 6680,
        '島根県': 6680,
        '岡山県': 6680,
        '広島県': 6680,
        '山口県': 6680,

        # Zone XIII (EPSG:6681) - 四国
        '香川県': 6681,
        '徳島県': 6681,
        '愛媛県': 6681,
        '高知県': 6681,

        # Zone XIV (EPSG:6682) - 九州北部
        '福岡県': 6682,
        '佐賀県': 6682,
        '長崎県': 6682,
        '大分県': 6682,

        # Zone XV (EPSG:6683) - 九州南部
        '熊本県': 6683,
        '宮崎県': 6683,
        '鹿児島県': 6683,

        # Zone XVI (EPSG:6684) - 冲绳
        '沖縄県': 6684,
    }

    return prefecture_epsg_map.get(pref, 6677)  # 默认使用东京的Zone IX

# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class GeocodingResult:
    """地理编码结果"""
    address: str
    latitude: float
    longitude: float
    confidence: str
    scenario: str
    method: str
    reasoning: str
    metadata: Dict

# ============================================================================
# 地址预处理
# ============================================================================

def normalize_address(address: str) -> str:
    """
    智能规范化地址字符串（基于KG地址层级结构）

    KG地址结构（空格分隔）：
    - 3部分：都道府県 + 市区 + 町村+番地 （罕见）
    - 4部分：都道府県 + 市区 + 町村 + 番地
    - 5部分：都道府県 + 市区 + 町村 + 丁目 + 番地

    处理：
    1. 统一连字符格式（全角/特殊符号 → 半角）
    2. 统一楼层后缀（F/f → 全角Ｆ）
    3. 自动在丁目前添加空格（确保符合5部分格式）
    4. 标准化空格（确保部分间只有一个空格）

    返回：符合KG标准格式的地址
    """
    # 去除首尾空格
    address = address.strip()

    # 字符规范化：统一连字符格式
    # - U+2212 (−) MINUS SIGN → U+002D (-) HYPHEN-MINUS
    # - U+FF0D (－) FULLWIDTH HYPHEN-MINUS → U+002D (-) HYPHEN-MINUS
    address = address.replace('\u2212', '-')  # − → -
    address = address.replace('\uff0d', '-')  # － → -

    # 统一楼层后缀为全角Ｆ（KG中存储的格式）
    address = re.sub(r'(\d+)F\b', r'\1Ｆ', address)  # 半角大写F → 全角Ｆ
    address = re.sub(r'(\d+)f\b', r'\1Ｆ', address)  # 半角小写f → 全角Ｆ

    # 关键：自动在丁目前添加空格（如果缺失）
    # 匹配模式：非空白字符 + 数字 + "丁目"
    # 例如：玉川3丁目 → 玉川 3丁目
    address = re.sub(r'([^\s\d])(\d+丁目)', r'\1 \2', address)

    # 标准化多个连续空格为单个空格
    address = re.sub(r'\s+', ' ', address)

    return address


def remove_floor_from_address(address: str) -> str:
    """
    从地址中移除楼层信息，生成基础地址

    支持的楼层格式：
    - 1-1-2Ｆ → 1-1
    - 1-1-301 → 1-1（三位数房间号）
    - 1-1-2階 → 1-1

    返回：不含楼层信息的基础地址
    """
    # 先规范化
    normalized = normalize_address(address)

    # 移除楼层后缀模式：
    # 模式1: 番地-楼层Ｆ (如 1-1-2Ｆ, 12-3-5Ｆ)
    base = re.sub(r'-\d+Ｆ\b', '', normalized)

    # 模式2: 番地-房间号（三位数，如 1-1-301）
    base = re.sub(r'-\d{3,}\b', '', base)

    # 模式3: 番地-楼层階 (如 1-1-2階)
    base = re.sub(r'-\d+階\b', '', base)

    return base


# ============================================================================
# 阶段一：地址解析和索引构建
# ============================================================================

def parse_address(address: str) -> Dict[str, str]:
    """
    结构化解析日本地址（支持全日本）

    支持的格式：
    1. 都+特别区: "東京都 板橋区 小豆沢 2丁目 22−15"
    2. 県+市+区: "静岡県 静岡市葵区 城内町 1-20" (政令指定市)
    3. 県+市: "静岡県 裾野市 千福が丘 1丁目 16-6" (普通市)
    4. 道府県+町/村: "北海道 虻田郡ニセコ町 中央通 60-4"

    输出: {pref, city, ward, town, chome, banchi_main, banchi_sub}
    注意: city, ward, chome, banchi_sub可能为None
    """
    parsed = {}

    # 预处理：规范化地址
    address = normalize_address(address)

    # 提取都道府県（必须包含都/道/府/県）
    pref_match = re.match(r'^(.+?[都道府県])\s+', address)
    if not pref_match:
        print(f"  ⚠️ 无法识别都道府県: {address}")
        return None

    parsed['pref'] = pref_match.group(1)
    remaining = address[len(parsed['pref']):].strip()

    # 格式1: 政令指定市（市+区结构，如 静岡市葵区）
    match = re.match(r'^(.+?市)(.+?区)\s+(.+?)\s+(\d+)丁目\s+(\d+)[−-](\d+)(?:[−-]\d+)?(?:.*)?$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = match.group(2)
        parsed['town'] = match.group(3)
        parsed['chome'] = match.group(4)
        parsed['banchi_main'] = match.group(5)
        parsed['banchi_sub'] = match.group(6)
        return parsed

    match = re.match(r'^(.+?市)(.+?区)\s+(.+?)\s+(\d+)丁目\s+(\d+)\s*$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = match.group(2)
        parsed['town'] = match.group(3)
        parsed['chome'] = match.group(4)
        parsed['banchi_main'] = match.group(5)
        parsed['banchi_sub'] = None
        return parsed

    match = re.match(r'^(.+?市)(.+?区)\s+(.+?)\s+(\d+)[−-](\d+)(?:[−-]\d+)?(?:.*)?$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = match.group(2)
        parsed['town'] = match.group(3)
        parsed['chome'] = None
        parsed['banchi_main'] = match.group(4)
        parsed['banchi_sub'] = match.group(5)
        return parsed

    match = re.match(r'^(.+?市)(.+?区)\s+(.+?)\s+(\d+)\s*$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = match.group(2)
        parsed['town'] = match.group(3)
        parsed['chome'] = None
        parsed['banchi_main'] = match.group(4)
        parsed['banchi_sub'] = None
        return parsed

    # 格式2: 特别区/普通区（东京都XX区，或单独的区）
    match = re.match(r'^([^市]+区)\s+(.+?)\s+(\d+)丁目\s+(\d+)[−-](\d+)(?:[−-]\d+)?(?:.*)?$', remaining)
    if match:
        parsed['city'] = None
        parsed['ward'] = match.group(1)
        parsed['town'] = match.group(2)
        parsed['chome'] = match.group(3)
        parsed['banchi_main'] = match.group(4)
        parsed['banchi_sub'] = match.group(5)
        return parsed

    match = re.match(r'^([^市]+区)\s+(.+?)\s+(\d+)丁目\s+(\d+)\s*$', remaining)
    if match:
        parsed['city'] = None
        parsed['ward'] = match.group(1)
        parsed['town'] = match.group(2)
        parsed['chome'] = match.group(3)
        parsed['banchi_main'] = match.group(4)
        parsed['banchi_sub'] = None
        return parsed

    match = re.match(r'^([^市]+区)\s+(.+?)\s+(\d+)[−-](\d+)(?:[−-]\d+)?(?:.*)?$', remaining)
    if match:
        parsed['city'] = None
        parsed['ward'] = match.group(1)
        parsed['town'] = match.group(2)
        parsed['chome'] = None
        parsed['banchi_main'] = match.group(3)
        parsed['banchi_sub'] = match.group(4)
        return parsed

    match = re.match(r'^([^市]+区)\s+(.+?)\s+(\d+)\s*$', remaining)
    if match:
        parsed['city'] = None
        parsed['ward'] = match.group(1)
        parsed['town'] = match.group(2)
        parsed['chome'] = None
        parsed['banchi_main'] = match.group(3)
        parsed['banchi_sub'] = None
        return parsed

    # 格式3: 普通市（市，无区）
    match = re.match(r'^(.+?市)\s+(.+?)\s+(\d+)丁目\s+(\d+)[−-](\d+)(?:[−-]\d+)?(?:.*)?$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = None
        parsed['town'] = match.group(2)
        parsed['chome'] = match.group(3)
        parsed['banchi_main'] = match.group(4)
        parsed['banchi_sub'] = match.group(5)
        return parsed

    match = re.match(r'^(.+?市)\s+(.+?)\s+(\d+)丁目\s+(\d+)\s*$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = None
        parsed['town'] = match.group(2)
        parsed['chome'] = match.group(3)
        parsed['banchi_main'] = match.group(4)
        parsed['banchi_sub'] = None
        return parsed

    match = re.match(r'^(.+?市)\s+(.+?)\s+(\d+)[−-](\d+)(?:[−-]\d+)?(?:.*)?$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = None
        parsed['town'] = match.group(2)
        parsed['chome'] = None
        parsed['banchi_main'] = match.group(3)
        parsed['banchi_sub'] = match.group(4)
        return parsed

    match = re.match(r'^(.+?市)\s+(.+?)\s+(\d+)\s*$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = None
        parsed['town'] = match.group(2)
        parsed['chome'] = None
        parsed['banchi_main'] = match.group(3)
        parsed['banchi_sub'] = None
        return parsed

    # 格式4: 町/村
    match = re.match(r'^(.+?[町村])\s+(.+?)\s+(\d+)丁目\s+(\d+)[−-](\d+)(?:[−-]\d+)?(?:.*)?$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = None
        parsed['town'] = match.group(2)
        parsed['chome'] = match.group(3)
        parsed['banchi_main'] = match.group(4)
        parsed['banchi_sub'] = match.group(5)
        return parsed

    match = re.match(r'^(.+?[町村])\s+(.+?)\s+(\d+)丁目\s+(\d+)\s*$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = None
        parsed['town'] = match.group(2)
        parsed['chome'] = match.group(3)
        parsed['banchi_main'] = match.group(4)
        parsed['banchi_sub'] = None
        return parsed

    match = re.match(r'^(.+?[町村])\s+(.+?)\s+(\d+)[−-](\d+)(?:[−-]\d+)?(?:.*)?$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = None
        parsed['town'] = match.group(2)
        parsed['chome'] = None
        parsed['banchi_main'] = match.group(3)
        parsed['banchi_sub'] = match.group(4)
        return parsed

    match = re.match(r'^(.+?[町村])\s+(.+?)\s+(\d+)\s*$', remaining)
    if match:
        parsed['city'] = match.group(1)
        parsed['ward'] = None
        parsed['town'] = match.group(2)
        parsed['chome'] = None
        parsed['banchi_main'] = match.group(3)
        parsed['banchi_sub'] = None
        return parsed

    # 所有格式都不匹配
    print(f"  ⚠️ 地址格式解析失败: {address}")
    return None


def build_indices(parsed: Dict[str, str]) -> Dict[str, str]:
    """
    构建KG查询索引（支持全日本地址结构）

    注意：
    - Ward和Block使用阿拉伯数字
    - Area使用汉字数字
    - 支持多种行政区划：都道府県/市/区/町/村
    """
    # 汉字数字映射
    kanji_map = {
        '1': '一', '2': '二', '3': '三', '4': '四', '5': '五',
        '6': '六', '7': '七', '8': '八', '9': '九', '10': '十'
    }

    indices = {}

    # 构建Ward索引（市区町村级别）
    if parsed.get('city') and parsed.get('ward'):
        # 政令指定市：县 + 市区（如 静岡県 静岡市葵区）
        indices['Ward'] = f"{parsed['pref']} {parsed['city']}{parsed['ward']}"
    elif parsed.get('ward'):
        # 特别区：都 + 区（如 東京都 板橋区）
        indices['Ward'] = f"{parsed['pref']} {parsed['ward']}"
    elif parsed.get('city'):
        # 普通市/町/村：県 + 市（如 静岡県 裾野市）
        indices['Ward'] = f"{parsed['pref']} {parsed['city']}"
    else:
        # 无法构建Ward索引
        indices['Ward'] = parsed['pref']

    # 构建Area和Block索引
    if parsed.get('chome') is not None:
        # 有丁目的情况
        chome_kanji = kanji_map.get(parsed['chome'], parsed['chome'])

        if parsed.get('city') and parsed.get('ward'):
            # 政令指定市
            indices['Area'] = f"{parsed['pref']} {parsed['city']}{parsed['ward']} {parsed['town']}{chome_kanji}丁目"
            indices['Block'] = f"{parsed['pref']} {parsed['city']}{parsed['ward']} {parsed['town']} {parsed['chome']}丁目 {parsed['banchi_main']}"
        elif parsed.get('ward'):
            # 特别区
            indices['Area'] = f"{parsed['pref']} {parsed['ward']} {parsed['town']}{chome_kanji}丁目"
            indices['Block'] = f"{parsed['pref']} {parsed['ward']} {parsed['town']} {parsed['chome']}丁目 {parsed['banchi_main']}"
        elif parsed.get('city'):
            # 普通市/町/村
            indices['Area'] = f"{parsed['pref']} {parsed['city']} {parsed['town']}{chome_kanji}丁目"
            indices['Block'] = f"{parsed['pref']} {parsed['city']} {parsed['town']} {parsed['chome']}丁目 {parsed['banchi_main']}"
    else:
        # 无丁目的情况
        if parsed.get('city') and parsed.get('ward'):
            # 政令指定市
            indices['Area'] = f"{parsed['pref']} {parsed['city']}{parsed['ward']} {parsed['town']}"
            indices['Block'] = f"{parsed['pref']} {parsed['city']}{parsed['ward']} {parsed['town']} {parsed['banchi_main']}"
        elif parsed.get('ward'):
            # 特别区
            indices['Area'] = f"{parsed['pref']} {parsed['ward']} {parsed['town']}"
            indices['Block'] = f"{parsed['pref']} {parsed['ward']} {parsed['town']} {parsed['banchi_main']}"
        elif parsed.get('city'):
            # 普通市/町/村
            indices['Area'] = f"{parsed['pref']} {parsed['city']} {parsed['town']}"
            indices['Block'] = f"{parsed['pref']} {parsed['city']} {parsed['town']} {parsed['banchi_main']}"

    return indices


# ============================================================================
# 阶段二：精确地址检索
# ============================================================================

def extract_base_address(address: str) -> str:
    """
    提取基础地址（忽略房间号/楼层信息）

    处理逻辑：
    1. 移除第三个连字符之后的所有内容（房间号通常是 XX-YY-ZZZ）
    2. 移除楼层标识（Ｆ、階等）
    3. 保留核心的都道府県-市区-町村-丁目-番地结构

    示例：
    - 東京都 世田谷区 玉川 3丁目 15-12-206 → 東京都 世田谷区 玉川 3丁目 15-12
    - 東京都 世田谷区 玉川 3丁目 15-12-2Ｆ → 東京都 世田谷区 玉川 3丁目 15-12
    - 東京都 世田谷区 玉川 3丁目 15-12 → 東京都 世田谷区 玉川 3丁目 15-12
    """
    # 分割地址为部分
    parts = address.split(' ')

    if len(parts) < 3:
        return address  # 地址太短，无法处理

    # 最后一部分是番地
    banchi = parts[-1]

    # 处理番地：保留前两个数字段（XX-YY），移除后续的房间号/楼层
    # 匹配模式：数字-数字，后面可能跟-数字或Ｆ等
    match = re.match(r'(\d+-\d+)', banchi)
    if match:
        base_banchi = match.group(1)
        parts[-1] = base_banchi
    else:
        # 没有副号的情况（如"相生町 21"），保留原样
        # 或者移除可能的楼层后缀
        parts[-1] = re.sub(r'-?\d*Ｆ$', '', banchi)  # 移除楼层Ｆ
        parts[-1] = re.sub(r'-?\d*階$', '', parts[-1])  # 移除楼层階

    return ' '.join(parts)


def exact_poi_retrieval(address: str) -> List[Dict]:
    """
    智能POI匹配（基于STARTS WITH策略）

    核心思想：
    - KG中的POI地址可能包含房间号/楼层信息（如 15-12-206）
    - 用户输入的基础地址（如 15-12）应该匹配所有 15-12* 的POI
    - 使用STARTS WITH而非精确匹配（=），避免遗漏带房间号的POI

    匹配策略：
    Level 1: STARTS WITH 完整输入地址（包含可能的房间号）
    Level 2: STARTS WITH 基础地址（去除房间号/楼层后）
    Level 3: STARTS WITH 更短的地址（去除番地部分，只匹配到丁目）

    返回：所有匹配的POI列表（后续会计算中位数坐标）
    """
    if CONFIG['verbose']:
        print(f"  [exact_poi_retrieval] 基于STARTS WITH的智能匹配")

    pois = []

    with neo4j_driver.session(database=os.getenv('NEO4J_DATABASE')) as session:

        # Level 1: STARTS WITH 完整输入地址
        if CONFIG['verbose']:
            print(f"  → Level 1: STARTS WITH完整地址 '{address}'")

        cypher_starts = """
        MATCH (p:POI)
        WHERE p.address STARTS WITH $address
        RETURN p.poi_id as poi_id,
               p.name as name,
               p.address as address,
               p.geometry as geometry
        """

        result = session.run(cypher_starts, address=address)
        for record in result:
            coords = parse_point_geometry(record['geometry'])
            if coords:
                pois.append({
                    'poi_id': record['poi_id'],
                    'name': record['name'],
                    'address': record['address'],
                    'latitude': coords[0],
                    'longitude': coords[1]
                })

        if pois:
            if CONFIG['verbose']:
                print(f"  ✓ Level 1 成功: 找到 {len(pois)} 个POI（包含所有房间号变体）")
            return pois

        # Level 2: STARTS WITH 基础地址（去除房间号/楼层）
        base_address = extract_base_address(address)

        if base_address != address:
            if CONFIG['verbose']:
                print(f"  ✗ Level 1 失败，尝试 Level 2: STARTS WITH基础地址 '{base_address}'")

            result = session.run(cypher_starts, address=base_address)
            for record in result:
                coords = parse_point_geometry(record['geometry'])
                if coords:
                    pois.append({
                        'poi_id': record['poi_id'],
                        'name': record['name'],
                        'address': record['address'],
                        'latitude': coords[0],
                        'longitude': coords[1]
                    })

            if pois:
                if CONFIG['verbose']:
                    print(f"  ✓ Level 2 成功: 找到 {len(pois)} 个POI")
                return pois

        # Level 3: STARTS WITH 到丁目级别（去除番地）
        parts = address.split(' ')
        if len(parts) >= 4:  # 至少有：都道府県 市区 町 番地
            # 保留到倒数第二部分（丁目或町村）
            chome_level = ' '.join(parts[:-1])

            if CONFIG['verbose']:
                print(f"  ✗ Level 2 失败，尝试 Level 3: STARTS WITH丁目级别 '{chome_level}'")

            cypher_fuzzy = """
            MATCH (p:POI)
            WHERE p.address STARTS WITH $address
            RETURN p.poi_id as poi_id,
                   p.name as name,
                   p.address as address,
                   p.geometry as geometry
            LIMIT 50
            """

            result = session.run(cypher_fuzzy, address=chome_level)
            for record in result:
                coords = parse_point_geometry(record['geometry'])
                if coords:
                    pois.append({
                        'poi_id': record['poi_id'],
                        'name': record['name'],
                        'address': record['address'],
                        'latitude': coords[0],
                        'longitude': coords[1]
                    })

            if pois:
                if CONFIG['verbose']:
                    print(f"  ✓ Level 3 成功: 找到 {len(pois)} 个POI（丁目级别匹配）")
            else:
                if CONFIG['verbose']:
                    print(f"  ✗ Level 3 失败: 未找到任何POI")

        return pois


def community_construction(matched_pois: List[Dict]) -> Dict:
    """
    社区构建：提取POI信息作为Reasoning Basis
    """
    return {
        'scenario': 'exact_match',
        'pois': matched_pois
    }


def backtracking_retrieval(block_address: str) -> Tuple[Optional[Dict], List[Dict]]:
    """
    回溯检索：查询Block实体和同Block的POI

    返回: (block_info, sibling_pois)
    """
    with neo4j_driver.session(database=os.getenv('NEO4J_DATABASE')) as session:

        # 查询Block实体
        block_cypher = """
        MATCH (b:Block)
        WHERE b.address = $block_address
        RETURN b.address as address,
               b.geometry as geometry,
               b.poi_count as poi_count
        """

        block_result = session.run(block_cypher, block_address=block_address)
        block_record = block_result.single()

        block_info = None
        if block_record:
            block_info = {
                'address': block_record['address'],
                'geometry': block_record['geometry'],
                'poi_count': block_record['poi_count']
            }

        # 查询同Block的POI
        poi_cypher = """
        MATCH (p:POI)-[:locatesAt]->(b:Block)
        WHERE b.address = $block_address
        RETURN p.poi_id as poi_id,
               p.name as name,
               p.address as address,
               p.geometry as geometry
        LIMIT 50
        """

        poi_result = session.run(poi_cypher, block_address=block_address)

        sibling_pois = []
        for record in poi_result:
            coords = parse_point_geometry(record['geometry'])
            if coords:
                sibling_pois.append({
                    'poi_id': record['poi_id'],
                    'name': record['name'],
                    'address': record['address'],
                    'latitude': coords[0],
                    'longitude': coords[1]
                })

        return block_info, sibling_pois


# ============================================================================
# 阶段三：空间约束算法前置（确定Seed）
# ============================================================================

def determine_seeds(block_info: Optional[Dict], sibling_pois: List[Dict],
                   parsed: Dict, indices: Dict) -> Tuple[List[Point], Dict]:
    """
    确定DCABG算法的种子点

    返回: (seeds, seed_metadata)
    """
    seeds = []

    # Case A: Block存在 + 有兄弟POI
    if block_info and sibling_pois:
        # 添加兄弟POI
        poi_ids = []
        for poi in sibling_pois:
            seeds.append(Point(poi['longitude'], poi['latitude']))
            poi_ids.append(poi['poi_id'])

        # 添加Block代表点（如果有）
        if block_info['geometry']:
            coords = parse_point_geometry(block_info['geometry'])
            if coords:
                seeds.append(Point(coords[1], coords[0]))  # (lon, lat)

        metadata = {
            'case': 'A',
            'description': 'Block exists with POIs and representative point',
            'seed_count': len(seeds),
            'source': 'sibling_pois + block_point',
            'poi_ids': poi_ids
        }

        return seeds, metadata

    # Case B: Block存在 + 有代表点 + 无兄弟POI
    elif block_info and block_info['geometry'] and not sibling_pois:
        coords = parse_point_geometry(block_info['geometry'])
        if coords:
            seeds.append(Point(coords[1], coords[0]))

        metadata = {
            'case': 'B',
            'description': 'Block exists with only representative point',
            'seed_count': len(seeds),
            'source': 'block_point_only',
            'poi_ids': []
        }

        return seeds, metadata

    # Case C: Block不存在 → Neighborhood Detection
    else:
        seeds, metadata = neighborhood_detection(parsed, indices)
        return seeds, metadata


def neighborhood_detection(parsed: Dict, indices: Dict,
                          max_distance: int = 10) -> Tuple[List[Point], Dict]:
    """
    邻域检测：双向渐进式搜索最近的番地主号
    """
    target_main = int(parsed['banchi_main'])

    with neo4j_driver.session(database=os.getenv('NEO4J_DATABASE')) as session:

        for distance in range(0, max_distance + 1):
            # 候选番地主号
            if distance == 0:
                candidates = [target_main]
            else:
                candidates = [target_main - distance, target_main + distance]

            for candidate_main in candidates:
                if candidate_main < 1:
                    continue

                # 构建邻居Block地址（支持多种行政区划结构）
                if parsed.get('chome') is not None:
                    # 有丁目
                    if parsed.get('city') and parsed.get('ward'):
                        neighbor_block_addr = f"{parsed['pref']} {parsed['city']}{parsed['ward']} {parsed['town']} {parsed['chome']}丁目 {candidate_main}"
                    elif parsed.get('ward'):
                        neighbor_block_addr = f"{parsed['pref']} {parsed['ward']} {parsed['town']} {parsed['chome']}丁目 {candidate_main}"
                    elif parsed.get('city'):
                        neighbor_block_addr = f"{parsed['pref']} {parsed['city']} {parsed['town']} {parsed['chome']}丁目 {candidate_main}"
                else:
                    # 无丁目
                    if parsed.get('city') and parsed.get('ward'):
                        neighbor_block_addr = f"{parsed['pref']} {parsed['city']}{parsed['ward']} {parsed['town']} {candidate_main}"
                    elif parsed.get('ward'):
                        neighbor_block_addr = f"{parsed['pref']} {parsed['ward']} {parsed['town']} {candidate_main}"
                    elif parsed.get('city'):
                        neighbor_block_addr = f"{parsed['pref']} {parsed['city']} {parsed['town']} {candidate_main}"

                # 查询邻居Block及其POI
                cypher = """
                MATCH (b:Block {address: $block_addr})
                OPTIONAL MATCH (p:POI)-[:locatesAt]->(b)
                RETURN b.address as block_address,
                       b.geometry as block_geometry,
                       collect(p) as pois
                """

                # cypher = """
                # MATCH (target:Block {address: $target_block_addr})-[:nearby]-(b:Block)
                # OPTIONAL MATCH (p:POI)-[:locatesAt]->(b)
                # RETURN b.address AS block_address,
                # b.geometry AS block_geometry,
                # collect(p) AS pois
                # """

                result = session.run(cypher, block_addr=neighbor_block_addr)
                record = result.single()

                if record and record['pois']:
                    # 找到了邻居Block
                    seeds = []

                    # 添加邻居POI
                    poi_ids = []
                    for poi in record['pois']:
                        if poi:  # 检查是否为null
                            coords = parse_point_geometry(poi['geometry'])
                            if coords:
                                seeds.append(Point(coords[1], coords[0]))
                                poi_ids.append(poi['poi_id'])

                    # 添加邻居Block代表点
                    if record['block_geometry']:
                        coords = parse_point_geometry(record['block_geometry'])
                        if coords:
                            seeds.append(Point(coords[1], coords[0]))

                    metadata = {
                        'case': 'C',
                        'description': f'Using neighbor Block {candidate_main} (distance={distance})',
                        'seed_count': len(seeds),
                        'source': f'neighbor_block_distance_{distance}',
                        'neighbor_block': neighbor_block_addr,
                        'poi_ids': poi_ids
                    }

                    return seeds, metadata

        # 如果仍未找到，使用Area级别的POI
        print(f"  ⚠️ 未找到邻居Block，使用Area级别POI")
        return query_area_pois(indices['Area'])


def build_area_name_candidates(area_name: str) -> List[str]:
    """
    构建Area名称候选（兼容空格/数字/汉字差异）
    """
    def strip_aza_suffix(name: str) -> str:
        # 去掉“字xxxx”尾缀以匹配官方町域名
        return re.sub(r'字.+$', '', name)

    def kanji_to_int(s: str) -> Optional[int]:
        if s.isdigit():
            return int(s)
        kanji_digit = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                       '六': 6, '七': 7, '八': 8, '九': 9}
        if s == '十':
            return 10
        if '十' in s:
            parts = s.split('十')
            if parts[0] == '':
                tens = 1
            else:
                tens = kanji_digit.get(parts[0])
            ones = kanji_digit.get(parts[1]) if len(parts) > 1 and parts[1] else 0
            if tens is None or ones is None:
                return None
            return tens * 10 + ones
        return kanji_digit.get(s)

    def int_to_kanji(n: int) -> Optional[str]:
        kanji_digit = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五',
                       6: '六', 7: '七', 8: '八', 9: '九'}
        if n == 10:
            return '十'
        if 1 <= n <= 9:
            return kanji_digit[n]
        if 11 <= n <= 99:
            tens, ones = divmod(n, 10)
            tens_part = kanji_digit.get(tens, '')
            ones_part = kanji_digit.get(ones, '')
            return f"{tens_part}十{ones_part}" if ones_part else f"{tens_part}十"
        return None

    def add_candidate(candidates: List[str], value: Optional[str]) -> None:
        if value and value not in candidates:
            candidates.append(value)

    candidates: List[str] = []
    tokens = area_name.split()
    if tokens:
        for i in range(len(tokens)):
            add_candidate(candidates, ' '.join(tokens[i:]))
    else:
        add_candidate(candidates, area_name)

    if tokens:
        if re.match(r'^(\d+|[一二三四五六七八九十]+)丁目$', tokens[-1]) and len(tokens) >= 2:
            # 末尾仅为“2丁目”这种形式
            short_name = f"{tokens[-2]} {tokens[-1]}"
            add_candidate(candidates, short_name)
            short_prefix = ' '.join(tokens[:-2])
        elif tokens[-1].endswith('丁目'):
            # 末尾已包含“町域+丁目”（无空格）
            short_name = tokens[-1]
            short_prefix = ' '.join(tokens[:-1])
        else:
            short_name = tokens[-1]
            short_prefix = ' '.join(tokens[:-1])
    else:
        short_name = area_name
        short_prefix = ''

    short_compact = short_name.replace(' ', '')
    m = re.match(r'^(.*?)(\d+|[一二三四五六七八九十]+)丁目$', short_compact)
    if m:
        town = m.group(1)
        chome_raw = m.group(2)
        chome_num = kanji_to_int(chome_raw)
        chome_kanji = int_to_kanji(chome_num) if chome_num is not None else None

        if chome_num is not None:
            add_candidate(candidates, f"{town}{chome_num}丁目")
            add_candidate(candidates, f"{town} {chome_num}丁目")
        if chome_kanji:
            add_candidate(candidates, f"{town}{chome_kanji}丁目")
            add_candidate(candidates, f"{town} {chome_kanji}丁目")

        if short_prefix:
            prefix = short_prefix
            for suffix in candidates[:]:
                if suffix == area_name:
                    continue
                add_candidate(candidates, f"{prefix} {suffix}")

    # 处理“字”尾缀（例如：宮加三字鳴沢 → 宮加三）
    for name in candidates[:]:
        stripped = strip_aza_suffix(name)
        if stripped != name:
            add_candidate(candidates, stripped)
            if short_prefix and ' ' not in stripped:
                prefix = short_prefix
                add_candidate(candidates, f"{prefix} {stripped}")

    return candidates


def query_area_pois(area_name: str) -> Tuple[List[Point], Dict]:
    """
    查询Area级别的POI（保底策略）
    """
    with neo4j_driver.session(database=os.getenv('NEO4J_DATABASE')) as session:
        cypher = """
        MATCH (a:Area)
        WHERE a.name IN $area_names
        MATCH (p:POI)-[:locatesAt]->(a)
        RETURN p.geometry as geometry, p.poi_id as poi_id
        LIMIT 20
        """

        area_candidates = build_area_name_candidates(area_name)
        result = session.run(cypher, area_names=area_candidates)

        seeds = []
        poi_ids = []
        for record in result:
            coords = parse_point_geometry(record['geometry'])
            if coords:
                seeds.append(Point(coords[1], coords[0]))
                poi_ids.append(record['poi_id'])

        metadata = {
            'case': 'C',
            'description': 'Using Area-level POIs (fallback)',
            'seed_count': len(seeds),
            'source': 'area_level_pois',
            'poi_ids': poi_ids
        }

        if not seeds:
            area_candidates = build_area_name_candidates(area_name)
            area_geom_query = """
            MATCH (a:Area)
            WHERE a.name IN $area_names
            RETURN a.geometry as geometry, a.name as name
            LIMIT 1
            """
            area_record = session.run(area_geom_query, area_names=area_candidates).single()
            if area_record and area_record['geometry']:
                area_geom = wkt.loads(area_record['geometry'])
                centroid = area_geom.centroid
                seeds.append(Point(centroid.x, centroid.y))
                metadata = {
                    'case': 'C',
                    'description': 'Using Area centroid (fallback)',
                    'seed_count': len(seeds),
                    'source': 'area_centroid',
                    'matched_area': area_record.get('name'),
                    'poi_ids': []
                }

        return seeds, metadata


# ============================================================================
# 阶段四：DCABG算法
# ============================================================================

def dcabg_algorithm(indices: Dict, seeds: List[Point], parsed: Dict,
                   seed_metadata: Optional[Dict] = None) -> Dict:
    """
    DCABG算法主函数

    返回: spatial_constraint
    """
    print(f"  [DCABG] 开始执行...")

    # Step 1: 查询Area边界
    P_area, town_code = query_area_polygon(indices['Area'])

    if not P_area:
        print(f"  ⚠️ Area数据缺失，使用简单凸包")
        return fallback_convex_hull(seeds)

    # Step 2: 优先使用POI附近的Road
    L_roads = []
    poi_ids = seed_metadata.get('poi_ids') if seed_metadata else None
    if poi_ids:
        L_roads = query_nearby_roads_for_pois(poi_ids)
        if L_roads:
            print(f"  [DCABG] 使用POI附近Road: {len(L_roads)}条")

    if not L_roads:
        L_roads = query_roads_by_town_code(town_code)

    if not L_roads:
        print(f"  ⚠️ Road数据缺失，使用简单凸包")
        return fallback_convex_hull(seeds)

    print(f"  [DCABG] Area边界: ✓, Roads: {len(L_roads)}条")

    # Step 2: 几何预处理与投影
    proj_area, proj_roads, proj_seeds = geometric_preprocessing(P_area, L_roads, seeds, parsed['pref'])

    # Step 3: 拓扑网络构建
    noded_lines = topological_network_construction(proj_area, proj_roads)

    # Step 4: 多边形化
    candidate_polygons = list(polygonize(noded_lines))
    print(f"  [DCABG] 候选多边形: {len(candidate_polygons)}个")

    if not candidate_polygons:
        print(f"  ⚠️ 多边形化失败，使用简单凸包")
        return fallback_convex_hull(seeds)

    # Step 5: 空间锚点选择
    target_polygon = spatial_anchoring(candidate_polygons, proj_seeds)

    if not target_polygon:
        print(f"  ⚠️ 空间锚点选择失败，使用简单凸包")
        return fallback_convex_hull(seeds)

    # Step 6: 验证与格式化
    epsg_code = get_epsg_for_prefecture(parsed['pref'])
    spatial_constraint = validation_and_formatting(target_polygon, epsg_code)

    print(f"  [DCABG] 完成! 面积: {spatial_constraint['area_m2']:.0f}m²")

    return spatial_constraint


def query_area_polygon(area_name: str) -> Tuple[Optional[Polygon], Optional[str]]:
    """
    查询Area边界并返回town_code
    """
    with neo4j_driver.session(database=os.getenv('NEO4J_DATABASE')) as session:

        # 查询Area
        area_cypher = """
        MATCH (a:Area)
        WHERE a.name = $area_name
        RETURN a.geometry as geometry, a.town_code as town_code
        """

        area_record = None
        area_candidates = build_area_name_candidates(area_name)
        for candidate in area_candidates:
            area_result = session.run(area_cypher, area_name=candidate)
            area_record = area_result.single()
            if area_record:
                break

        if not area_record:
            print(f"  ⚠️ 未找到Area: {area_name}")
            return None, None

        P_area = wkt.loads(area_record['geometry'])
        town_code = area_record['town_code']

        return P_area, town_code


def query_roads_by_town_code(town_code: Optional[str]) -> List[LineString]:
    """
    根据town_code查询Road
    """
    if not town_code:
        return []

    with neo4j_driver.session(database=os.getenv('NEO4J_DATABASE')) as session:
        road_cypher = """
        MATCH (a:Area {town_code: $town_code})<-[:in]-(r:Road)
        RETURN r.geometry as geometry
        LIMIT 200
        """

        road_result = session.run(road_cypher, town_code=town_code)

        L_roads = []
        for record in road_result:
            road_geom = wkt.loads(record['geometry'])
            if isinstance(road_geom, LineString):
                L_roads.append(road_geom)

        return L_roads


def query_nearby_roads_for_pois(poi_ids: List[int]) -> List[LineString]:
    """
    使用POI附近Road作为优先道路集合
    """
    if not poi_ids:
        return []

    with neo4j_driver.session(database=os.getenv('NEO4J_DATABASE')) as session:
        road_cypher = """
        MATCH (p:POI)-[:nearby]->(r:Road)
        WHERE p.poi_id IN $poi_ids
        RETURN DISTINCT r.geometry as geometry
        LIMIT 200
        """

        road_result = session.run(road_cypher, poi_ids=poi_ids)

        L_roads = []
        for record in road_result:
            road_geom = wkt.loads(record['geometry'])
            if isinstance(road_geom, LineString):
                L_roads.append(road_geom)

        return L_roads


def query_area_and_roads(area_name: str) -> Tuple[Optional[Polygon], List[LineString]]:
    """
    查询Area边界和Road网络（兼容旧调用）
    """
    P_area, town_code = query_area_polygon(area_name)
    if not P_area:
        return None, []
    L_roads = query_roads_by_town_code(town_code)
    return P_area, L_roads


def geometric_preprocessing(P_area: Polygon, L_roads: List[LineString],
                            seeds: List[Point], pref: str) -> Tuple:
    """
    几何预处理与投影 (WGS84 → JGD2011)
    根据都道府县自动选择合适的平面直角坐标系
    """
    # 根据都道府县选择EPSG代码
    epsg_code = get_epsg_for_prefecture(pref)

    # 定义投影转换器
    project_to_meter = Transformer.from_crs(
        'EPSG:4326',  # WGS84
        f'EPSG:{epsg_code}',  # JGD2011平面直角坐标系
        always_xy=True
    ).transform

    # 投影
    proj_area = transform(project_to_meter, P_area)
    proj_roads = [transform(project_to_meter, road) for road in L_roads]
    proj_seeds = [transform(project_to_meter, seed) for seed in seeds]

    # 数据清理
    proj_roads = [r for r in proj_roads if r.length > 0]

    return proj_area, proj_roads, proj_seeds


def topological_network_construction(proj_area: Polygon,
                                     proj_roads: List[LineString]) -> MultiLineString:
    """
    拓扑网络构建：边界集成 + Unary Union
    """
    # 提取Area边界
    area_boundary = proj_area.boundary

    # 集成所有线段
    all_lines = proj_roads + [area_boundary]

    # Unary Union（节点化）
    noded_lines = unary_union(all_lines)

    return noded_lines


def spatial_anchoring(candidate_polygons: List[Polygon],
                     proj_seeds: List[Point]) -> Optional[Polygon]:
    """
    空间锚点选择
    """
    if not proj_seeds:
        # 无种子点（理论上不会发生）
        print(f"  ⚠️ 无种子点，返回最大多边形")
        return max(candidate_polygons, key=lambda p: p.area)

    if len(proj_seeds) == 1:
        # Case B: 单个种子点
        for poly in candidate_polygons:
            if poly.contains(proj_seeds[0]):
                return poly

    else:
        # Case A/C: 多个种子点
        seed_counts = {}
        for poly in candidate_polygons:
            count = sum(1 for seed in proj_seeds if poly.contains(seed))
            seed_counts[poly] = count

        # 选择包含最多种子的多边形
        if seed_counts:
            return max(seed_counts, key=seed_counts.get)

    return None


def validation_and_formatting(target_polygon: Polygon, epsg_code: int) -> Dict:
    """
    验证与格式化
    """
    # 合理性检查
    area_m2 = target_polygon.area

    if not (50 < area_m2 < 10000):
        print(f"  ⚠️ 面积异常: {area_m2:.0f}m²")

    # 坐标系还原 (JGD2011 → WGS84)
    reproject_to_wgs84 = Transformer.from_crs(
        f'EPSG:{epsg_code}',
        'EPSG:4326',
        always_xy=True
    ).transform

    P_virtual = transform(reproject_to_wgs84, target_polygon)

    # 特征提取
    centroid = P_virtual.centroid
    bounds = P_virtual.bounds

    # 计算半径
    radius_m = calculate_max_radius(P_virtual)

    return {
        'polygon': P_virtual,
        'centroid': (centroid.x, centroid.y),  # (lon, lat)
        'area_m2': area_m2,
        'bounds': bounds,
        'radius_m': radius_m
    }


def calculate_max_radius(polygon: Polygon) -> float:
    """
    计算多边形的最大半径（从质心到顶点）
    """
    from math import radians, cos, sin, sqrt, atan2

    centroid = polygon.centroid
    center_lon, center_lat = centroid.x, centroid.y

    max_dist = 0
    for coord in polygon.exterior.coords:
        lon, lat = coord

        # Haversine距离
        R = 6371000  # 地球半径（米）
        phi1 = radians(center_lat)
        phi2 = radians(lat)
        delta_phi = radians(lat - center_lat)
        delta_lambda = radians(lon - center_lon)

        a = sin(delta_phi/2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        dist = R * c
        max_dist = max(max_dist, dist)

    return max_dist


def fallback_convex_hull(seeds: List[Point]) -> Dict:
    """
    降级策略：简单凸包
    """
    if not seeds:
        return None

    from shapely.geometry import MultiPoint

    if len(seeds) < 3:
        # 点太少，创建缓冲区
        center = seeds[0]
        buffer_polygon = center.buffer(0.001)  # 约100米

        return {
            'polygon': buffer_polygon,
            'centroid': (center.x, center.y),
            'area_m2': 10000,  # 估算
            'bounds': buffer_polygon.bounds,
            'radius_m': 100
        }

    multi_point = MultiPoint(seeds)
    convex_hull = multi_point.convex_hull

    return {
        'polygon': convex_hull,
        'centroid': (convex_hull.centroid.x, convex_hull.centroid.y),
        'area_m2': convex_hull.area * 10000,  # 粗略估算
        'bounds': convex_hull.bounds,
        'radius_m': calculate_max_radius(convex_hull)
    }


# ============================================================================
# 阶段五：LLM推理
# ============================================================================

def llm_reasoning(address: str, community_result: Optional[Dict],
                 spatial_constraint: Optional[Dict], seeds: List[Point],
                 seed_metadata: Dict) -> GeocodingResult:
    """
    LLM推理主函数
    """
    # 路径1: 有精确匹配
    if community_result:
        return llm_reasoning_exact_match(address, community_result)

    # 路径2: 无精确匹配，使用DCABG
    else:
        return llm_reasoning_with_dcabg(address, spatial_constraint, seeds, seed_metadata)


def llm_reasoning_exact_match(address: str, community_result: Dict) -> GeocodingResult:
    """
    路径1: 基于精确匹配的POI推理
    """
    pois = community_result['pois']

    # 构建Prompt
    prompt = f"""You are a professional Japanese geocoding expert. Please return the most appropriate coordinates based on the matched POIs.

【Target Address】
{address}

【Matched POIs】
"""

    for idx, poi in enumerate(pois, 1):
        prompt += f"\n{idx}. {poi['name']}"
        prompt += f"\n   Address: {poi['address']}"
        prompt += f"\n   Coordinates: ({poi['latitude']:.6f}, {poi['longitude']:.6f})"

    prompt += """

【Task】
Select the most likely coordinates from the POIs above, or calculate the average coordinates.

Please return in JSON format:
{
  "latitude": <latitude_value>,
  "longitude": <longitude_value>,
  "reasoning": "<reasoning_for_selection>",
  "confidence": "high"
}

Return only JSON, no other text.
"""

    # 调用LLM
    response = call_llm(prompt, temperature=0.1)
    result = parse_llm_response(response)

    if not result:
        # LLM解析失败，使用平均值
        avg_lat = sum(p['latitude'] for p in pois) / len(pois)
        avg_lon = sum(p['longitude'] for p in pois) / len(pois)

        return GeocodingResult(
            address=address,
            latitude=avg_lat,
            longitude=avg_lon,
            confidence='high',
            scenario='one_to_one' if len(pois) == 1 else 'one_to_many',
            method='exact_match',
            reasoning=f"Using average coordinates of {len(pois)} matched POI(s)",
            metadata={'matched_pois': len(pois), 'poi_names': [p['name'] for p in pois]}
        )

    return GeocodingResult(
        address=address,
        latitude=result['latitude'],
        longitude=result['longitude'],
        confidence=result.get('confidence', 'high'),
        scenario='one_to_one' if len(pois) == 1 else 'one_to_many',
        method='exact_match',
        reasoning=result.get('reasoning', 'Based on exact matched POI'),
        metadata={'matched_pois': len(pois), 'poi_names': [p['name'] for p in pois]}
    )


def llm_reasoning_with_dcabg(address: str, spatial_constraint: Dict,
                             seeds: List[Point], seed_metadata: Dict) -> GeocodingResult:
    """
    路径2: 基于DCABG空间约束的推理
    """

    prompt = f"""You are a professional Japanese geocoding expert. Please infer the precise coordinates of the target address based on spatial constraints and reference points.

【Target Address】
{address}

【Spatial Constraint Boundary】(Constructed from actual road network)
- Center Point: ({spatial_constraint['centroid'][0]:.6f}, {spatial_constraint['centroid'][1]:.6f})
- Constraint Radius: {spatial_constraint['radius_m']:.1f} meters
- Polygon Area: {spatial_constraint['area_m2']:.0f} m²
- Boundary Bounds: {spatial_constraint['bounds']}
- **Important**: The inferred result must be within this boundary

【Reference Points (Seeds)】({seed_metadata['description']})
Total {seed_metadata['seed_count']} reference points:
"""

    for idx, seed in enumerate(seeds[:10], 1): 
        prompt += f"\n{idx}. Seed Point {idx}: ({seed.y:.6f}, {seed.x:.6f})"

    prompt += f"""

【Japanese Address Numbering Patterns】
1. Banchi numbers typically increase along roads
2. Sub-numbers (go) are distributed within the same Block
3. Larger sub-numbers are usually farther from the Block center

【Reasoning Requirements】
1. Reference the distribution of Seed points
2. Consider address numbering patterns
3. Infer the target location should be within the spatial constraint boundary
4. The returned coordinates should be close to the constraint center point

Please return in JSON format:
{{
  "latitude": <latitude_value>,
  "longitude": <longitude_value>,
  "reasoning": "<reasoning_process>",
  "confidence": "medium"
}}

Return only JSON, no other text.
"""

    # 调用LLM
    response = call_llm(prompt, temperature=0.2)
    result = parse_llm_response(response)

    if not result:
        # LLM解析失败，使用约束中心点
        return GeocodingResult(
            address=address,
            latitude=spatial_constraint['centroid'][1],
            longitude=spatial_constraint['centroid'][0],
            confidence='low',
            scenario='zero_match',
            method='DCABG + fallback',
            reasoning="LLM parsing failed, using constraint center point",
            metadata=seed_metadata
        )

    # 边界验证
    predicted_point = Point(result['longitude'], result['latitude'])
    P_virtual = spatial_constraint['polygon']

    final_lat = result['latitude']
    final_lon = result['longitude']
    validation_status = "in_boundary"

    if not P_virtual.contains(predicted_point):
        # 投影到最近的边界点
        closest = nearest_points(predicted_point, P_virtual.boundary)[1]
        final_lat = closest.y
        final_lon = closest.x
        validation_status = "projected_to_boundary"
        result['reasoning'] += " [Projected to boundary]"

    metadata = {
        **seed_metadata,
        'spatial_constraint': {
            'area_m2': spatial_constraint['area_m2'],
            'radius_m': spatial_constraint['radius_m'],
            'centroid': spatial_constraint['centroid']
        },
        'validation_status': validation_status
    }

    return GeocodingResult(
        address=address,
        latitude=final_lat,
        longitude=final_lon,
        confidence=result.get('confidence', 'medium'),
        scenario='zero_match',
        method='DCABG + LLM interpolation',
        reasoning=result.get('reasoning', 'Based on DCABG constraint reasoning'),
        metadata=metadata
    )


def call_llm(prompt: str, temperature: float = 0.1) -> str:
    """
    调用OpenAI API
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional geocoding expert specializing in precise coordinate inference based on spatial constraints and reference points."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=400
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  ⚠️ LLM调用失败: {str(e)}")
        return None


def parse_llm_response(response: str) -> Optional[Dict]:
    """
    解析LLM的JSON响应
    """
    if not response:
        return None

    try:
        # 提取JSON部分
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except Exception as e:
        print(f"  ⚠️ JSON解析失败: {str(e)}")

    return None


# ============================================================================
# 工具函数
# ============================================================================

def parse_point_geometry(geom_str: str) -> Optional[Tuple[float, float]]:
    """
    解析POINT几何字符串

    输入: "POINT (139.678901 35.776543)"
    输出: (35.776543, 139.678901)  # (lat, lon)
    """
    try:
        point = wkt.loads(geom_str)
        return (point.y, point.x)  # (lat, lon)
    except:
        return None


# ============================================================================
# 主工作流
# ============================================================================

def geocode_address(address: str) -> GeocodingResult:
    """
    完整的地理编码工作流
    """
    # 预处理：规范化地址（统一字符格式）
    address = normalize_address(address)

    print(f"\n{'='*80}")
    print(f"处理地址: {address}")
    print(f"{'='*80}")

    # ========================================================================
    # 阶段一: 地址解析和索引构建
    # ========================================================================
    print("\n[阶段一] 地址解析和索引构建...")

    parsed = parse_address(address)
    if not parsed:
        return GeocodingResult(
            address=address,
            latitude=0.0,
            longitude=0.0,
            confidence='failed',
            scenario='parse_error',
            method='none',
            reasoning='Address parsing failed',
            metadata={}
        )

    indices = build_indices(parsed)
    print(f"  Ward: {indices['Ward']}")
    print(f"  Area: {indices['Area']}")
    print(f"  Block: {indices['Block']}")

    # ========================================================================
    # 阶段二: 精确地址检索
    # ========================================================================
    print("\n[阶段二] 精确地址检索...")

    matched_pois = exact_poi_retrieval(address)

    if matched_pois:
        print(f"  ✓ 找到 {len(matched_pois)} 个精确匹配的POI")
        community_result = community_construction(matched_pois)

        # 直接跳转到阶段五
        print("\n[阶段五] LLM推理（精确匹配路径）...")
        result = llm_reasoning(address, community_result, None, [], {})
        return result

    print(f"  未找到精确匹配，进入回溯检索...")

    # Phase 2: Backtracking Retrieval
    block_info, sibling_pois = backtracking_retrieval(indices['Block'])

    if block_info:
        print(f"  ✓ 找到Block实体，POI数量: {len(sibling_pois)}")
    else:
        print(f"  Block实体不存在")

    # ========================================================================
    # 阶段三: 空间约束算法前置（确定Seed）
    # ========================================================================
    print("\n[阶段三] 确定Seed点...")

    seeds, seed_metadata = determine_seeds(block_info, sibling_pois, parsed, indices)
    print(f"  Case: {seed_metadata['case']}")
    print(f"  Seed数量: {seed_metadata['seed_count']}")
    print(f"  来源: {seed_metadata['source']}")

    if not seeds:
        print(f"  ⚠️ 无法获取Seed点，使用降级策略")
        return GeocodingResult(
            address=address,
            latitude=0.0,
            longitude=0.0,
            confidence='failed',
            scenario='extreme_fallback',
            method='no_reference_points',
            reasoning='Unable to obtain any reference points',
            metadata=seed_metadata
        )

    # ========================================================================
    # 阶段四: DCABG算法
    # ========================================================================
    print("\n[阶段四] DCABG算法...")

    spatial_constraint = dcabg_algorithm(indices, seeds, parsed, seed_metadata)

    if not spatial_constraint:
        print(f"  ⚠️ DCABG算法失败")
        return GeocodingResult(
            address=address,
            latitude=seeds[0].y if seeds else 0.0,
            longitude=seeds[0].x if seeds else 0.0,
            confidence='low',
            scenario='dcabg_failed',
            method='seed_fallback',
            reasoning='DCABG failed, using first seed point',
            metadata=seed_metadata
        )

    # ========================================================================
    # 阶段五: LLM推理
    # ========================================================================
    print("\n[阶段五] LLM推理（DCABG路径）...")

    result = llm_reasoning(address, None, spatial_constraint, seeds, seed_metadata)

    return result


# ============================================================================
# CSV输出
# ============================================================================

def save_results_to_csv(results: List[GeocodingResult], output_file: str):
    """
    保存结果到CSV
    """
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow([
            'address',
            'latitude',
            'longitude',
            'confidence',
            'scenario',
            'method',
            'reasoning',
            'metadata'
        ])

        # 写入数据
        for result in results:
            writer.writerow([
                result.address,
                f"{result.latitude:.6f}",
                f"{result.longitude:.6f}",
                result.confidence,
                result.scenario,
                result.method,
                result.reasoning,
                json.dumps(result.metadata, ensure_ascii=False)
            ])

    print(f"\n✓ 结果已保存到: {output_file}")


# ============================================================================
# 主函数
# ============================================================================

def load_addresses_from_config() -> List[str]:
    """
    根据CONFIG配置加载地址列表
    """
    mode = CONFIG["input_mode"]

    if mode == "test":
        # 测试模式：使用预定义地址
        addresses = CONFIG["test_addresses"]
        print(f"📝 输入模式: 测试模式 ({len(addresses)} 个预定义地址)")

    elif mode == "file":
        # 文件模式：从CSV读取
        input_file = CONFIG["input_file"]
        address_column = CONFIG["address_column"]

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        addresses = []
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if address_column in row:
                    address = row[address_column].strip()
                    if address:  # 跳过空地址
                        addresses.append(address)
                else:
                    raise KeyError(f"CSV文件中未找到列: {address_column}")

        print(f"📂 输入模式: 文件模式")
        print(f"   文件路径: {input_file}")
        print(f"   地址列名: {address_column}")
        print(f"   总地址数: {len(addresses)}")

    else:
        raise ValueError(f"不支持的输入模式: {mode}，请使用 'test' 或 'file'")

    # 应用数量限制
    limit = CONFIG.get("limit")
    if limit is not None and limit > 0:
        original_count = len(addresses)
        addresses = addresses[:limit]
        print(f"⚠️  已限制处理数量: {len(addresses)}/{original_count}")

    return addresses


def is_failure_result(result: GeocodingResult) -> bool:
    """
    判断结果是否应计为失败。
    """
    if result.confidence in ['failed', 'very_low']:
        return True
    if result.scenario in ['extreme_fallback', 'parse_error']:
        return True
    return False


def main():
    """
    主函数
    """
    print("="*80)
    print("KG-Enhanced Geocoding with DCABG Algorithm")
    print("="*80)
    print()

    # 加载地址列表
    try:
        addresses = load_addresses_from_config()
    except Exception as e:
        print(f"❌ 加载地址失败: {str(e)}")
        return

    if not addresses:
        print("⚠️  没有要处理的地址")
        return

    print(f"\n{'='*80}")
    print(f"开始处理 {len(addresses)} 个地址")
    print(f"{'='*80}")

    results = []
    success_count = 0
    failure_count = 0

    for idx, address in enumerate(addresses, 1):
        try:
            if CONFIG["verbose"]:
                print()  # 空行分隔
            else:
                # 简洁模式：只显示进度
                print(f"\r[{idx}/{len(addresses)}] 处理中: {address[:50]}...", end='', flush=True)

            result = geocode_address(address)
            results.append(result)

            # 根据置信度判断成功或失败
            if is_failure_result(result):
                failure_count += 1
            else:
                success_count += 1

            # 根据配置决定是否显示结果
            if CONFIG["show_results"]:
                print(f"\n{'='*80}")
                print(f"【结果 {idx}/{len(addresses)}】")
                print(f"  地址: {result.address}")
                print(f"  坐标: ({result.latitude:.6f}, {result.longitude:.6f})")
                print(f"  置信度: {result.confidence}")
                print(f"  场景: {result.scenario}")
                print(f"  方法: {result.method}")
                if CONFIG["verbose"]:
                    print(f"  推理: {result.reasoning[:150]}...")
                print(f"{'='*80}")

        except Exception as e:
            failure_count += 1
            print(f"\n❌ 处理失败 [{idx}/{len(addresses)}]: {address}")
            print(f"   错误: {str(e)}")
            if CONFIG["verbose"]:
                import traceback
                traceback.print_exc()

    # 打印换行（如果之前是简洁模式）
    if not CONFIG["verbose"]:
        print()

    # 保存结果
    print(f"\n{'='*80}")
    print("保存结果...")
    print(f"{'='*80}")

    output_file = CONFIG["output_file"]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    save_results_to_csv(results, output_file)

    # 关闭连接
    neo4j_driver.close()

    # 显示统计信息
    print(f"\n{'='*80}")
    print("处理完成！")
    print(f"{'='*80}")
    print(f"总地址数: {len(addresses)}")
    print(f"成功: {success_count}")
    print(f"失败: {failure_count}")
    print(f"成功率: {success_count/len(addresses)*100:.1f}%")
    print(f"结果已保存到: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
