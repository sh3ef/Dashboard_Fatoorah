import streamlit as st
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.datapipeline import DataPipeline

def main():
    # إنشاء وتشغيل خط المعالجة
    pipeline = DataPipeline()
    pipeline.run_pipeline()
    
    # إعداد صفحة Streamlit
    st.set_page_config(layout="wide")
    st.title('📊 لوحة تحليل المخزون والمبيعات (كاملة)')
    
    # عرض الرسوم البيانية
    col1, col2, col3 = st.columns(3)
    col1.plotly_chart(pipeline.visualizations['fig1'], use_container_width=True)
    col2.plotly_chart(pipeline.visualizations['fig2'], use_container_width=True)
    col3.plotly_chart(pipeline.visualizations['fig5'], use_container_width=True)

    col4, col5 = st.columns(2)
    col4.plotly_chart(pipeline.visualizations['fig3'], use_container_width=True)
    col5.plotly_chart(pipeline.visualizations['fig4'], use_container_width=True)

    col6, col7 = st.columns(2)
    col6.plotly_chart(pipeline.visualizations['fig6'], use_container_width=True)
    col7.plotly_chart(pipeline.visualizations['fig7'], use_container_width=True)

    col8, col9 = st.columns(2)
    col8.plotly_chart(pipeline.visualizations['fig9'], use_container_width=True)
    col9.plotly_chart(pipeline.visualizations['fig10'], use_container_width=True)

    col10, col11 = st.columns(2)
    col10.plotly_chart(pipeline.visualizations['fig11'], use_container_width=True)
    col11.plotly_chart(pipeline.visualizations['fig12'], use_container_width=True)

    col12, col13 = st.columns(2)
    col12.plotly_chart(pipeline.visualizations['fig13'], use_container_width=True)
    col13.plotly_chart(pipeline.visualizations['fig15'], use_container_width=True)

    col14, col15 = st.columns(2)
    col14.plotly_chart(pipeline.visualizations['fig14'], use_container_width=True)
    col15.plotly_chart(pipeline.visualizations['fig16'], use_container_width=True)

    st.plotly_chart(pipeline.visualizations['fig17'], use_container_width=True)
    
    # جدول المخزون الحالي
    st.header("📦 المخزون الحالي")
    current_stock = pipeline.get_current_stock_table()
    st.dataframe(
        current_stock[['product_id', 'name', 'current_stock']]
        .sort_values('current_stock', ascending=False)
        .style.background_gradient(subset='current_stock', cmap='Greens'),
        height=400,
        column_config={
            "product_id": "ID المنتج",
            "name": "اسم المنتج", 
            "current_stock": "المخزون الحالي"
        }
    )

if __name__ == "__main__":
    main()