import pandas as pd
import logging
from typing import List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def excel_to_list(file_path: str) -> List[str]:
    """
    将Excel文件转换为字符串列表
    
    Args:
        file_path: Excel文件路径
        
    Returns:
        字符串列表
    """
    try:
        # 读取Excel文件，指定引擎为openpyxl
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # 将每行转换为字符串
        result = []
        for _, row in df.iterrows():
            # 将行转换为字符串，忽略NaN值
            row_str = ' '.join(str(val) for val in row if pd.notna(val))
            if row_str.strip():  # 只添加非空行
                result.append(row_str)
                
        return result
        
    except Exception as e:
        logger.error(f"处理Excel文件时出错: {str(e)}")
        raise

def list_to_excel(result_map: Dict[str, List[Any]], output_file: str) -> None:
    """
    将招标参数对比结果转换为Excel文件
    
    Args:
        result_map: 招标参数到对比结果列表的映射
        output_file: 输出Excel文件路径
    """
    try:
        # 准备数据
        data = []
        for requirement, results in result_map.items():
            if len(results) >= 5:
                satisfaction_status = results[0]
                detailed_analysis = results[1]
                source1 = results[2]
                source2 = results[3]
                source3 = results[4]
            else:
                # 兼容旧格式
                satisfaction_status = results[0] if results else '信息不足'
                detailed_analysis = results[1] if len(results) > 1 else ''
                source1 = results[2] if len(results) > 2 else ''
                source2 = results[3] if len(results) > 3 else ''
                source3 = results[4] if len(results) > 4 else ''
                
            data.append({
                '招标要求': requirement,
                '满足情况': satisfaction_status,
                '详细分析': detailed_analysis,
                '依据文档1': source1,
                '依据文档2': source2,
                '依据文档3': source3
            })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        
        # 设置列宽以便更好地显示内容
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='招标参数对比结果')
            
            # 获取工作表并设置列宽
            worksheet = writer.sheets['招标参数对比结果']
            worksheet.column_dimensions['A'].width = 30  # 招标要求
            worksheet.column_dimensions['B'].width = 15  # 满足情况
            worksheet.column_dimensions['C'].width = 50  # 详细分析
            worksheet.column_dimensions['D'].width = 40  # 依据文档1
            worksheet.column_dimensions['E'].width = 40  # 依据文档2
            worksheet.column_dimensions['F'].width = 40  # 依据文档3
            
            # 设置自动换行
            from openpyxl.styles import Alignment
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
        
        logger.info(f"招标参数对比结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"保存Excel文件时出错: {str(e)}")
        raise