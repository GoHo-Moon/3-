import streamlit as st
import pandas as pd
import sys
import os

# --------------------------------------------------------
# [강제 경로 설정] 이 코드를 넣으면 무조건 해결됩니다.
# 현재 main.py가 있는 폴더 위치를 찾아서 파이썬에게 알려줌
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# --------------------------------------------------------

import api  # 이제 무조건 찾습니다.

# ... (아래 set_page_config 부터는 기존 코드 그대로) ...
# 1. 페이지 설정
st.set_page_config(
    page_title="이슈 파인더",
    page_icon="🔍",
    layout="wide"
)

# 2. 데이터 수집 함수 (캐싱)
@st.cache_data
def fetch_news_data(keyword, num):
    return api.get_naver_news(keyword, num)

# 3. 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    st.info("메인 페이지입니다.")

# 4. 메인 화면
st.title("🗣️ 소셜 미디어 여론 분석")
st.divider()

col1, col2 = st.columns([4, 1])
with col1:
    keyword = st.text_input("검색어 입력", placeholder="예: 서울시 부동산")
with col2:
    st.write("") 
    st.write("") 
    search_btn = st.button("수집 시작", use_container_width=True)

# 5. 실행
if search_btn:
    if not keyword:
        st.warning("검색어를 입력하세요!")
    else:
        with st.spinner("수집 중..."):
            try:
                df = fetch_news_data(keyword, 1000)
                if not df.empty:
                    st.session_state['news_df'] = df
                    st.session_state['search_keyword'] = keyword
                    st.success(f"완료! {len(df)}개")
                else:
                    st.warning("결과 없음")
            except Exception as e:
                st.error(f"에러: {e}")

# 6. 결과 표출
if 'news_df' in st.session_state and not st.session_state['news_df'].empty:
    with st.expander("데이터 확인", expanded=True):
        st.dataframe(st.session_state['news_df'])