"""
prompt.py - 旅游风格生成的提示模板模块

该模块包含：
1. 各种旅游风格生成的提示模板
2. 提示格式化函数
3. 提示模板管理类
"""

from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class TravelTrajectory:
    """旅游轨迹数据结构"""
    user_id: str
    locations: List[str]
    timestamps: List[str]
    activities: List[str]
    region: str  # 'hometown' or 'destination'

class TravelPromptTemplates:
    """旅游风格生成提示模板类"""
    
    @staticmethod
    def hometown_to_destination_prompt(locations_str: str, activities_str: str, 
                                     num_locations: int, target_region: str) -> str:
        """
        基于家乡轨迹预测目的地旅游风格的提示模板
        
        Args:
            locations_str: 地点列表字符串
            activities_str: 活动列表字符串
            num_locations: 地点数量
            target_region: 目标地区
            
        Returns:
            格式化的提示文本
        """
        return f"""Based on the user's travel behavior in their hometown, predict their travel style in the destination.

Hometown travel trajectory:
- Visited locations: {locations_str}
- Activities: {activities_str}
- Travel pattern: {num_locations} locations visited

Please describe the user's travel style in {target_region} in 2-3 sentences, focusing on:
1. Preferred types of attractions
2. Activity preferences
3. Travel pace and schedule

Travel style prediction for {target_region}:"""

    @staticmethod
    def destination_actual_style_prompt(locations_str: str, activities_str: str, 
                                      num_locations: int) -> str:
        """
        基于目的地实际轨迹描述旅游风格的提示模板
        
        Args:
            locations_str: 地点列表字符串
            activities_str: 活动列表字符串
            num_locations: 地点数量
            
        Returns:
            格式化的提示文本
        """
        return f"""Based on the user's actual travel behavior in the destination, describe their travel style.

Destination travel trajectory:
- Visited locations: {locations_str}
- Activities: {activities_str}
- Travel pattern: {num_locations} locations visited

Please describe the user's actual travel style in 2-3 sentences, focusing on:
1. Types of attractions visited
2. Activity preferences shown
3. Travel pace and schedule observed

Actual travel style:"""

    @staticmethod
    def detailed_travel_analysis_prompt(locations_str: str, activities_str: str,
                                      timestamps_str: str, region: str, target_region: str = None) -> str:
        """
        详细的旅游行为分析提示模板
        
        Args:
            locations_str: 地点列表字符串
            activities_str: 活动列表字符串
            timestamps_str: 时间戳列表字符串
            region: 当前地区类型
            target_region: 目标地区（仅在预测时使用）
            
        Returns:
            格式化的提示文本
        """
        if region == "hometown" and target_region:
            return f"""Analyze the user's travel behavior in their hometown and predict their detailed travel preferences for {target_region}.

Hometown Travel Analysis:
- Locations visited: {locations_str}
- Activities engaged: {activities_str}
- Visit times: {timestamps_str}

Based on this hometown behavior, predict the user's travel style in {target_region}, including:
1. Preferred attraction types (cultural, natural, entertainment, shopping, etc.)
2. Activity preferences (active/passive, indoor/outdoor, social/solitary)
3. Travel rhythm (fast-paced/leisurely, early/late schedules)
4. Exploration style (planned/spontaneous, popular/hidden gems)

Detailed travel style prediction for {target_region}:"""
        else:
            return f"""Analyze the user's actual travel behavior in the destination and provide a comprehensive travel style description.

Destination Travel Analysis:
- Locations visited: {locations_str}
- Activities engaged: {activities_str}
- Visit times: {timestamps_str}

Provide a detailed analysis of the user's actual travel style, including:
1. Attraction preferences demonstrated (cultural, natural, entertainment, shopping, etc.)
2. Activity patterns observed (active/passive, indoor/outdoor, social/solitary)
3. Travel rhythm shown (fast-paced/leisurely, early/late schedules)
4. Exploration approach (planned/spontaneous, popular/hidden gems)

Comprehensive travel style analysis:"""

    @staticmethod
    def comparative_travel_style_prompt(hometown_locations: str, hometown_activities: str,
                                      destination_locations: str, destination_activities: str,
                                      target_region: str) -> str:
        """
        比较家乡和目的地旅游风格的提示模板
        
        Args:
            hometown_locations: 家乡地点字符串
            hometown_activities: 家乡活动字符串
            destination_locations: 目的地地点字符串
            destination_activities: 目的地活动字符串
            target_region: 目标地区
            
        Returns:
            格式化的提示文本
        """
        return f"""Compare the user's travel behavior between their hometown and destination, then analyze their travel style consistency.

Hometown Travel:
- Locations: {hometown_locations}
- Activities: {hometown_activities}

{target_region} Destination Travel:
- Locations: {destination_locations}
- Activities: {destination_activities}

Please analyze:
1. Similarities in travel preferences between hometown and destination
2. Differences and adaptations in travel behavior
3. Core travel style characteristics that remain consistent
4. How the user adapts their preferences to new environments

Travel style consistency analysis:"""

class TravelPromptFormatter:
    """旅游提示格式化器"""
    
    def __init__(self):
        self.templates = TravelPromptTemplates()
    
    def format_trajectory_data(self, trajectory: TravelTrajectory) -> Dict[str, str]:
        """
        格式化轨迹数据为字符串
        
        Args:
            trajectory: 旅游轨迹数据
            
        Returns:
            格式化后的字符串字典
        """
        return {
            "locations_str": ", ".join(trajectory.locations),
            "activities_str": ", ".join(trajectory.activities),
            "timestamps_str": ", ".join(trajectory.timestamps),
            "num_locations": len(trajectory.locations)
        }
    
    def trajectory_to_prompt(self, trajectory: TravelTrajectory, target_region: str, 
                           prompt_type: str = "basic") -> str:
        """
        将旅游轨迹转换为LLM输入提示
        
        Args:
            trajectory: 旅游轨迹数据
            target_region: 目标地区（用于生成风格的地区）
            prompt_type: 提示类型 ('basic', 'detailed', 'comparative')
            
        Returns:
            格式化的提示文本
        """
        formatted_data = self.format_trajectory_data(trajectory)
        
        if prompt_type == "basic":
            if trajectory.region == "hometown":
                return self.templates.hometown_to_destination_prompt(
                    formatted_data["locations_str"],
                    formatted_data["activities_str"],
                    formatted_data["num_locations"],
                    target_region
                )
            else:
                return self.templates.destination_actual_style_prompt(
                    formatted_data["locations_str"],
                    formatted_data["activities_str"],
                    formatted_data["num_locations"]
                )
        
        elif prompt_type == "detailed":
            return self.templates.detailed_travel_analysis_prompt(
                formatted_data["locations_str"],
                formatted_data["activities_str"],
                formatted_data["timestamps_str"],
                trajectory.region,
                target_region if trajectory.region == "hometown" else None
            )
        
        else:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
    
    def create_comparative_prompt(self, hometown_trajectory: TravelTrajectory, 
                                destination_trajectory: TravelTrajectory,
                                target_region: str) -> str:
        """
        创建比较家乡和目的地旅游风格的提示
        
        Args:
            hometown_trajectory: 家乡旅游轨迹
            destination_trajectory: 目的地旅游轨迹
            target_region: 目标地区
            
        Returns:
            比较分析提示
        """
        hometown_data = self.format_trajectory_data(hometown_trajectory)
        destination_data = self.format_trajectory_data(destination_trajectory)
        
        return self.templates.comparative_travel_style_prompt(
            hometown_data["locations_str"],
            hometown_data["activities_str"],
            destination_data["locations_str"],
            destination_data["activities_str"],
            target_region
        )

class CustomPromptBuilder:
    """自定义提示构建器"""
    
    def __init__(self):
        self.base_templates = {
            "prediction": "Based on the user's travel behavior in their {source_region}, predict their travel style in {target_region}.",
            "analysis": "Analyze the user's travel behavior and describe their travel style.",
            "comparison": "Compare travel behaviors and identify consistent travel patterns."
        }
        self.focus_areas = [
            "Preferred types of attractions",
            "Activity preferences", 
            "Travel pace and schedule",
            "Exploration style",
            "Social interaction patterns",
            "Budget and spending patterns"
        ]
    
    def build_custom_prompt(self, template_type: str, trajectory_data: Dict[str, str],
                          focus_areas: Optional[List[str]] = None,
                          additional_context: Optional[str] = None) -> str:
        """
        构建自定义提示
        
        Args:
            template_type: 模板类型
            trajectory_data: 轨迹数据
            focus_areas: 关注领域列表
            additional_context: 额外上下文
            
        Returns:
            自定义提示文本
        """
        if template_type not in self.base_templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        # 构建基础提示
        base_prompt = self.base_templates[template_type]
        
        # 添加轨迹信息
        trajectory_section = f"""
Travel trajectory:
- Visited locations: {trajectory_data.get('locations_str', 'N/A')}
- Activities: {trajectory_data.get('activities_str', 'N/A')}
- Travel pattern: {trajectory_data.get('num_locations', 0)} locations visited"""
        
        if trajectory_data.get('timestamps_str'):
            trajectory_section += f"\n- Visit times: {trajectory_data['timestamps_str']}"
        
        # 添加关注领域
        if focus_areas is None:
            focus_areas = self.focus_areas[:4]  # 默认使用前4个
        
        focus_section = "\nPlease focus on:\n"
        for i, area in enumerate(focus_areas, 1):
            focus_section += f"{i}. {area}\n"
        
        # 组合提示
        full_prompt = base_prompt + trajectory_section + focus_section
        
        if additional_context:
            full_prompt += f"\nAdditional context: {additional_context}\n"
        
        full_prompt += "\nTravel style description:"
        
        return full_prompt

# 预定义的特殊场景提示
class SpecialScenarioPrompts:
    """特殊场景提示模板"""
    
    @staticmethod
    def cultural_adaptation_prompt(hometown_culture: str, destination_culture: str,
                                 trajectory_data: Dict[str, str]) -> str:
        """跨文化旅游适应性分析提示"""
        return f"""Analyze how the user adapts their travel style when moving from {hometown_culture} culture to {destination_culture} culture.

Travel behavior:
- Locations: {trajectory_data['locations_str']}
- Activities: {trajectory_data['activities_str']}

Consider cultural differences in:
1. Social interaction expectations
2. Dining and food exploration habits
3. Religious and cultural site visits
4. Shopping and bargaining behaviors
5. Transportation preferences

Cultural adaptation analysis:"""

    @staticmethod
    def seasonal_travel_prompt(season: str, trajectory_data: Dict[str, str]) -> str:
        """季节性旅游行为分析提示"""
        return f"""Analyze the user's travel behavior during {season} season and predict their seasonal travel preferences.

{season.title()} travel behavior:
- Locations: {trajectory_data['locations_str']}
- Activities: {trajectory_data['activities_str']}
- Times: {trajectory_data.get('timestamps_str', 'N/A')}

Consider seasonal factors:
1. Weather-dependent activity preferences
2. Indoor vs outdoor attraction choices
3. Seasonal event participation
4. Travel timing and duration patterns

Seasonal travel style analysis:"""

    @staticmethod
    def budget_conscious_prompt(price_indicators: List[str], trajectory_data: Dict[str, str]) -> str:
        """预算意识旅游分析提示"""
        price_str = ", ".join(price_indicators)
        return f"""Analyze the user's budget-conscious travel behavior and spending patterns.

Travel behavior with budget indicators:
- Locations: {trajectory_data['locations_str']}
- Activities: {trajectory_data['activities_str']}
- Price indicators: {price_str}

Analyze budget preferences:
1. Free vs paid attraction preferences
2. Dining budget patterns
3. Transportation cost considerations
4. Shopping and souvenir spending
5. Accommodation type preferences

Budget-conscious travel analysis:"""

# 示例使用函数
def get_sample_prompts():
    """获取示例提示"""
    # 创建示例轨迹
    sample_trajectory = TravelTrajectory(
        user_id="user_001",
        locations=["Central Park", "Metropolitan Museum", "Times Square"],
        timestamps=["09:00", "14:00", "18:00"],
        activities=["walking", "sightseeing", "shopping"],
        region="hometown"
    )
    
    # 创建格式化器
    formatter = TravelPromptFormatter()
    
    # 生成不同类型的提示
    basic_prompt = formatter.trajectory_to_prompt(sample_trajectory, "Paris", "basic")
    detailed_prompt = formatter.trajectory_to_prompt(sample_trajectory, "Paris", "detailed")
    
    return {
        "basic": basic_prompt,
        "detailed": detailed_prompt
    }

if __name__ == "__main__":
    # 演示提示生成
    prompts = get_sample_prompts()
    print("=== Basic Prompt ===")
    print(prompts["basic"])
    print("\n=== Detailed Prompt ===")
    print(prompts["detailed"])
