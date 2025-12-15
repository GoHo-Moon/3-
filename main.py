import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 폰트 매니저
from wordcloud import WordCloud
from konlpy.tag import Okt
from collections import Counter
import networkx as nx
from itertools import combinations
import seaborn as sns
import altair as alt
import plotly.express as px

import api  # 네이버 뉴스 API 호출 함수 (별도 파일에 구현 가정)

# ----------------------------------------------------------------------
# A. 초기 설정 및 폰트 전역 등록
# ----------------------------------------------------------------------

# 폰트 경로 설정 (WordCloud와 Matplotlib 모두 사용)

# LLM을 사용,,, seaborn 및 networkx에서 font_path만으로는 실행이 오류가 났음.
# LLM 코드 참조
FONT_PATH = "./fonts/AppleSDGothicNeoB.ttf"
FONT_NAME = 'sans-serif' # 기본값
if os.path.exists(FONT_PATH):
    # 1. Matplotlib에 폰트 등록
    try:
        fm.fontManager.addfont(FONT_PATH)
        # 2. 등록된 폰트 이름 가져와서 설정
        FONT_NAME = fm.FontProperties(fname=FONT_PATH).get_name()
        plt.rc('font', family=FONT_NAME)
        plt.rc('axes', unicode_minus=False) # 마이너스 기호 깨짐 방지
    except Exception as e:
        # 등록 실패 시 Windows 기본 폰트 사용 
        st.warning(f"폰트 등록 오류: {e}. 기본 폰트로 대체됩니다.")
        FONT_NAME = 'Malgun Gothic'
        plt.rc('font', family=FONT_NAME)
        plt.rc('axes', unicode_minus=False)
else:
    st.warning(f"한글 폰트 파일이 없어 기본 폰트로 출력됩니다. ({FONT_PATH} 확인 필요)")
    # 파일이 없을 경우 Matplotlib 기본 폰트 설정 유지

# ======================================================
# 1) 페이지 설정 (
# ======================================================
st.set_page_config(
    page_title="🔥K팝 데몬 헌터스 팬덤 형성 핵심 요인 분석🔥",
    page_icon="🔍",
    layout="wide"
)

# ======================================================
# 2) 상단 고정 헤더
# ======================================================
st.markdown(
    """
    <div style="padding:14px 18px; border-radius:10px; border:1px solid #ddd; background:#fafafa;">
        <div style="font-size:18px; font-weight:700;">
            C221082 문현율
        </div>
        <div style="font-size:28px; font-weight:800; margin-top:6px;">
            🔥K팝 데몬 헌터스 팬덤 형성 핵심 요인 분석🔥
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")
st.info("🔥K팝 데몬 헌터스 팬덤 형성 핵심 요인 분석🔥")

# ----------------------------------------------------------------------
# B. [캐시 함수] 데이터 수집 
# 캐시 함수를 통해 같은 파라미터로 여러 번 호출 시 API 호출을 줄임..!
# ----------------------------------------------------------------------
@st.cache_data
def fetch_news_data(keyword, num):
    return api.get_naver_news(keyword, num)

# ----------------------------------------------------------------------
# C. [캐시 함수] 불용어 로드 (파일 IO는 한 번만)
# wordcloud와 networkx 분석 모두에서 동일한 불용어 리스트 사용..
# 미리 정의 후 캐싱 --> 캐싱을 통해 여러 번 호출 시 부담을 줄임.
# 강의안 보고 사용.
# ----------------------------------------------------------------------
@st.cache_data
def get_stop_words(keyword):
    """불용어 파일을 읽고 검색어 및 강의에서 사용된 불용어를 추가하여 반환합니다."""
    stopwords_path = "./data/korean_stopwords.txt" # 강의록에서 가지고온 불용어
    stop_words = set()
    
    if os.path.exists(stopwords_path):
        with open(stopwords_path, "r", encoding="utf-8") as f:
            stop_words = set(line.strip() for line in f if line.strip())
    else:
        # 기본 불용어 (파일 없을 경우.. 그럴 리는 없다.)
        stop_words = {"것", "등", "위", "수", "배", "만", "명", "관련", "대해", "뉴스", "속보"}
        
    if keyword:
        stop_words.add(keyword)
        stop_words.add(keyword.replace(" ", ""))
    # 강의안에서 추가된 불용어들 (중복 방지를 위해 set에 update)
    stop_words.update([
        "서울", "서울시", "부동산", "주요", "첫째", "결과", "조사", "아크", "대비", "증권", 
        "가능성", "대표", "시절", "제자", "최강", "활용", "최진", "타운", "요소", "적용",
        "중앙", "전주", "한국", "포함", "도시", "일부", "이슈", "보고서", "갈등", "미래", 
        "위원", "통해", "문제", "NH투자증권", "아유경제_부동산", "quot", "조국", 
        "조희연", "사면", "심층분석", "년", "월", "일", "시" # 시간 관련 불용어 추가(시계열 분석을 위해)
    ])
    return stop_words

# ----------------------------------------------------------------------
# D. [캐시 함수] 통합 분석 (형태소 분석은 한 번만)
# 마찬가지로 wordcloud와 networkx 분석 모두에서 동일한 형태소 분석 결과 사용..
# LLM의 힘을 빌려, 최적화를 진행했다. (95%그대로 사용..)
# ----------------------------------------------------------------------
@st.cache_data
def analyze_data(df, keyword, min_len):
    """
    데이터프레임을 분석하여 단어 빈도(freq)와 엣지 목록(edge_list)을 반환합니다.
    @st.cache_data를 사용하여 무거운 형태소 분석을 캐싱합니다.
    """
    stop_words = get_stop_words(keyword)
    
    # 텍스트 통합 (시리즈로 처리)
    text_series = (df["title"].fillna("").astype(str) + " " + df["description"].fillna("").astype(str))
    
    okt = Okt()
    all_filtered_nouns = []
    all_edge_list = []
    
    # 각 문서별로 처리
    for doc in text_series:
        # 1. 명사 추출
        nouns = okt.nouns(doc)
        
        # 2. 필터링 (최소 길이, 불용어)
        filtered_nouns = [n for n in nouns if len(n) >= min_len and n not in stop_words]
        
        # 3. 워드클라우드용: 모든 문서의 명사를 합칩니다.
        all_filtered_nouns.extend(filtered_nouns)
        
        # 4. 네트워크용: 동시 등장 관계 생성
        # 4-1. 중복 제거된 단어 리스트 (네트워크 분석에선 단어와의 관계를 보기에 중복된 단어는 set을 통해 하나로 취급.. LLM의 조언)
        unique_terms = sorted(set(filtered_nouns))
        # 4-2. 단어 쌍 생성 (조합)
        if len(unique_terms) >= 2: # 단어가 2개 이상일 때만 조합 생성
            all_edge_list.extend(combinations(unique_terms, 2)) # 엣지 리스트에 추가


    # 5. 최종 단어 빈도 계산
    freq = Counter(all_filtered_nouns)

    return freq, all_edge_list
# ----------------------------------------------------------------------
# E. [캐시 함수] 시계열 분석 데이터 전처리
# plotly, seaborn, altair 시각화에 공통으로 사용되는 시계열 데이터 전처리
# pubDate를 기준으로 일별 기사 건수 및 상위 N개 키워드의 일별 등장 빈도 계산..!

# ----------------------------------------------------------------------
@st.cache_data
def get_time_series_data(df, freq, min_len, top_n=5):
    """
    일별 기사 건수 및 상위 N개 키워드의 일별 등장 빈도를 계산합니다.
    """
    # pubDate를 날짜(Date) 형식으로 변환하고 날짜만 남깁니다.
    # pubDate 칼럼은 이미 데이터 수집 단계에서 datetime 객체로 변환되었다고 가정합니다.
    df_copy = df.copy() # 원본 데이터프레임 보호
    df_copy['date'] = pd.to_datetime(df_copy['pubDate']).dt.date
    df_copy['datetime'] = pd.to_datetime(df_copy['pubDate']) # 요일
    
    # 1. 일별 기사 건수 (Plotly용)
    daily_volume = df_copy.groupby('date').size().reset_index(name='기사_건수')
    
    # 요일 컬럼 추가 및 한글 변환
    daily_volume['datetime'] = pd.to_datetime(daily_volume['date'])
    daily_volume['요일'] = daily_volume['datetime'].dt.day_name(locale='ko_KR.utf-8')
    #요일 정보 업데이트 -> llm 참고.
    # 요일 순서를 강제하여 그래프 순서가 월, 화, 수... 순으로 되도록 카테고리 설정
    day_order = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    daily_volume['요일'] = pd.Categorical(daily_volume['요일'], categories=day_order, ordered=True)
    daily_volume = daily_volume.sort_values(['date']) # 날짜 순 정렬

    # 2. 상위 N개 키워드 목록 (Altair 시계열 추적 대상)
    top_words = [word for word, count in freq.most_common(top_n)]
    
    # 3. 일별 키워드 빈도 (Altair용)
    all_time_series_data = []
    okt = Okt()
    
    # 불용어 목록 로드 (캐시 함수 호출)
    # min_len 변수는 analyze_data에서 사용되므로 여기서는 직접 사용하지 않습니다.
    stop_words = get_stop_words(st.session_state.get("search_keyword", "")) 

    for date, group_df in df_copy.groupby('date'):
        # 해당 일자의 모든 텍스트 결합
        # (title + description)
        daily_text = " ".join(group_df["title"].fillna("").astype(str) + " " + group_df["description"].fillna("").astype(str))
        
        # 형태소 분석 및 필터링
        nouns = okt.nouns(daily_text)
        # min_len과 stop_words를 적용하여 필터링
        daily_nouns = [n for n in nouns if len(n) >= min_len and n not in stop_words] 
        daily_freq = Counter(daily_nouns)
        
        for word in top_words:
            all_time_series_data.append({
                '날짜': date,
                '단어': word,
                '빈도': daily_freq.get(word, 0)
            })
    
    # [수정] for 루프 밖에서 DataFrame을 생성해야 합니다.
    time_series_df = pd.DataFrame(all_time_series_data) 
    
    return daily_volume, time_series_df.sort_values('날짜')
# ======================================================
# 3) 사이드바 (인터렉티브한 조작 구현~)
# ======================================================
with st.sidebar:
    st.header("⚙️ 분석 설정")
    keyword = st.text_input(
        "검색어 입력",
        placeholder="예: K팝 데몬 헌터스",
        key="search_keyword_input"
    )
    news_limit = st.slider(
        "수집 기사 수", 
        min_value=100,
        max_value=1000,
        value=1000,
        step=100
    )
    st.subheader("📊 시각화 옵션")
    wc_top_n = st.slider(
        "워드클라우드 단어 수 (Top N)",
        min_value=20,
        max_value=200,
        value=80,
        step=10
    )
    edge_top_n = st.slider(
        "네트워크 관계 수 (Top N)",
        min_value=10,
        max_value=100,
        value=50,
        step=10
    )
    min_word_len = st.slider(
        "단어 최소 길이",
        min_value=1,
        max_value=4,
        value=2
    )
    # [추가] 시계열 Top N 설정
    ts_top_n = st.slider(
        "시계열 단어 수 (Top N)",
        min_value=1,
        max_value=10,
        value=5
    )
    search_btn = st.button("수집 시작", use_container_width=True)

# ======================================================
# 4) 메인 화면 – 데이터 수집 실행
# ======================================================
st.header("1. 데이터 수집")
if search_btn:
    if not keyword:
        st.warning("검색어를 입력하세요.")
    else:
        with st.spinner("뉴스 데이터 수집 중..."):
            try:
                # 데이터 수집 (캐싱 함수 사용)
                df = fetch_news_data(keyword, news_limit)
                
                if df is not None and not df.empty:
                    # st.session_state에 수집된 데이터와 검색어 저장
                    st.session_state["news_df"] = df
                    st.session_state["search_keyword"] = keyword
                    st.success(f"수집 완료: {len(df)}건")
                else:
                    st.warning("검색 결과가 없습니다.")
                    st.session_state["news_df"] = pd.DataFrame() # 빈 DF로 초기화
            except Exception as e:
                st.error(f"에러 발생: {e}")

# ======================================================
# 5) 데이터 확인
# ======================================================
if "news_df" in st.session_state and not st.session_state["news_df"].empty:
    st.header("2. 수집 데이터 확인")
    with st.expander("뉴스 데이터 보기"):
        st.dataframe(st.session_state["news_df"])

# ======================================================
# 6) 통합 분석 실행 및 데이터 준비
# ======================================================
if "news_df" in st.session_state and not st.session_state["news_df"].empty:
    df = st.session_state["news_df"]
    keyword = st.session_state.get("search_keyword", "")
    
    with st.spinner("통합 텍스트 분석 중 (형태소 분석 및 관계 생성)..."):
        # [핵심] 통합 분석 함수 호출 (캐시 함수)
        try:
            freq, edge_list = analyze_data(df, keyword, min_word_len)
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            freq = Counter()
            edge_list = []

    if not freq:
        st.warning("분석 가능한 명사가 없어 워드클라우드/네트워크를 생성할 수 없습니다. (단어 최소 길이 조절 필요)")
    daily_volume, time_series_df = get_time_series_data(df, freq, min_word_len, ts_top_n)
# ======================================================
# 7) 워드클라우드 시각화
# ======================================================
st.header("3. 워드클라우드 시각화")
if "news_df" in st.session_state and not st.session_state["news_df"].empty and freq:
    
    # 1. 워드클라우드 생성
    wc = WordCloud(
        font_path=FONT_PATH if os.path.exists(FONT_PATH) else None,
        background_color="white",
        width=900,
        height=450,
        max_words=int(wc_top_n)
    ).generate_from_frequencies(
        dict(freq.most_common(int(wc_top_n)))
    )
    
    # 2. 시각화 출력
    fig = plt.figure(figsize=(12, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig, clear_figure=True)
    
    # 3. 단어 빈도 표
    with st.expander("단어 빈도 Top 50"):
        st.dataframe(
            pd.DataFrame(
                freq.most_common(50),
                columns=["단어", "빈도"]
            ),
            use_container_width=True
        )

    st.info("""

            **💡 워드클라우드 해석 **
            * **노드(단어) 크기**:  등장 빈도를 나타냅니다, K팝 데몬 헌터스 분석를 분석 한 경우,
            가장 많이 뜬 단어는 케이팝, 데몬, 헌터스, 넷플릭스, 케데헌, 애니메이션 등등 으로
            이를 통해 k팝 데몬 헌터스는 넷플릭스와 관련이 깊다는 것, 케이팝에 관련된 애니라는 것을 알 수 있습니다.
    """)
# ======================================================
# 8) 네트워크 시각화
# ======================================================
st.header("4. 키워드 네트워크 분석")

if "news_df" in st.session_state and not st.session_state["news_df"].empty and edge_list:

    # 1. 엣지 빈도 계산
    edge_counts = Counter(edge_list)
    
    # 2. 상위 N개 엣지 필터링
    top_edges = edge_counts.most_common(int(edge_top_n))

    if len(top_edges) == 0:
        st.warning(f"상위 {int(edge_top_n)}개 관계를 찾을 수 없습니다. (설정 조절 필요)")
    else:
        # 3. 그래프 객체 생성
        G = nx.Graph()
        weighted_edges = [(u, v, weight) for (u, v), weight in top_edges]
        G.add_weighted_edges_from(weighted_edges)
        
        # 4. 중심성 계산 (Degree Centrality: 노드 크기 결정)
        centrality = nx.degree_centrality(G)
        
        # 5. 시각화 준비
        fig, ax = plt.subplots(figsize=(15, 15)) 
        
        # 레이아웃 결정 (힘 기반 배치)
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
        # 노드 크기 및 엣지 두께 설정
        node_size = [v * 7000 for v in centrality.values()]
        edge_width = [d['weight'] * 0.05 for (u, v, d) in G.edges(data=True)]

        # 6. draw_networkx 호출
        nx.draw_networkx(
            G, 
            pos,
            with_labels=True,
            node_size=node_size,
            node_color="#63B8FF", # 산뜻한 색상으로 변경
            edge_color="gray",
            width=edge_width,
            font_family=FONT_NAME, # 전역 설정된 폰트 사용
            font_size=12,
            alpha=0.8,
            ax=ax
        )
        
        ax.set_title(f"'{keyword}' 키워드 관계망 (Top {int(edge_top_n)})", size=20)
        plt.axis('off')
        st.pyplot(fig, clear_figure=True)
        
        st.info(f"""
        **💡 시각화 해석 가이드 (총 {len(G.nodes())}개 노드)**
        * [cite_start]**노드(단어) 크기**: 연결 중심성) , 즉, 다른 단어들과 얼마나 많이 직접적으로 연결되었는지.
        * [cite_start]**선(Edge) 두께**: 동시 등장 빈도 (관계의 강도). 두 단어가 기사에서 함께 나온 횟수를 의미합니다.
        * [cite_start]**레이아웃**: 힘 기반 배치. 관계가 강한 단어일수록 서로 가깝게 배치됩니다.

        결론 : 관계가 깊은 쌍들은 '케이팝 데몬 헌터스'내의 단어들과, 애니메이션, 케이팝, 영화 등등으로
        '케이팝 데몬 헌터스'가 케이팝과 애니메이션, 영화 등과 깊은 관련이 있음을 알 수 있습니다.
        """)
        
        # 순서쌍으로 조금 더 쉽게 보는 기능 (숨기기 가능=expander)
        with st.expander("상위 관계 목록"):
            st.dataframe(
                pd.DataFrame(
                    top_edges,
                    columns=["단어 쌍", "빈도"]
                ),
                use_container_width=True
            )

# ======================================================
# 9) Seaborn: 분석 기간 중 상위 단어 빈도 막대 그래프
# ======================================================
st.header("5. 단어 빈도 막대 그래프 (Seaborn)")
if "news_df" in st.session_state and not st.session_state["news_df"].empty and freq:
    
    top_n_freq_df = pd.DataFrame(
        freq.most_common(int(ts_top_n)), 
        columns=["단어", "빈도"]
    )

    if not top_n_freq_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        # 수평 막대 그래프 (빈도 순)
        sns.barplot(
            x="빈도", 
            y="단어", 
            data=top_n_freq_df.sort_values(by="빈도", ascending=False), 
            ax=ax,
            palette="viridis" # 색상 팔레트 지정
        )
        ax.set_title(f"키워드 빈도 Top {int(ts_top_n)}", size=15)
        ax.set_xlabel("빈도", size=12)
        ax.set_ylabel("단어", size=12)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
    else:
        st.warning("막대 그래프를 생성할 충분한 데이터가 없습니다.")
    
    st.info(f"""
        **💡 시각화 해석 가이드 
             * **막대 그래프**: 각 단어의 빈도를 시각적으로 비교할 수 있습니다.
             * **색상**: 단어의 중요도를 나타내며, 더 진한 색일수록 높은 빈도를 의미합니다.
             * **레이아웃**: 수평 막대 그래프는 단어의 상대적인 빈도를 쉽게 파악할 수 있게 도와줍니다.
            
            top 10으로 햇을 때는 데몬, 헌터스 ,케이팝,,,, 넷플릭스 애니메이션, 영화, 올해, 인기, 미국'
            으로 케이팝 데몬 헌터스는 올해 인기 있는 영화/애니/넷플릭스로서 특히 미국에도 인기가 많다는 것을 유추할 수 있다.
            """)
# ======================================================
# 10) Plotly: 일별 뉴스 발행 건수 시계열 그래프
# ======================================================
st.header("6. 일별 뉴스 발행량 추이 (Plotly)")
if "news_df" in st.session_state and not st.session_state["news_df"].empty:
    
    if not daily_volume.empty:
        # Plotly Express를 이용한 시계열 꺾은선 그래프 (추세를 보기 좋으니)
        fig = px.line(
            daily_volume,
            x='date',
            y='기사_건수',
            title='일별 기사 발행 건수 변화 추이',
            labels={'date': '날짜', '기사_건수': '기사 건수'},
            line_shape='linear',  # 꺾은선 그래프.
            markers=True,         # 각 데이터 포인트에 마커(점) 표시
            color_discrete_sequence=['#1F77B4'] # 파란색 계열로 변경 (선 그래프에 일반적) # 색은 llm한테 물어봄
        )
        
        # 레이아웃 업데이트 (시계열 최적화)
        fig.update_xaxes(
            tickformat="%Y-%m-%d", 
            title='날짜'
        )
        fig.update_layout(
            xaxis_title='날짜', 
            yaxis_title='기사 건수',
            hovermode="x unified" # 마우스 오버 시 x축 기준으로 통합 툴팁 표시
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("일별 뉴스 발행 건수 데이터를 찾을 수 없습니다.")
    st.info("""
            **💡 시각화 해석 가이드**)
            **꺾은선 그래프**: 시간에 따른 뉴스 발행량의 변화를 시각적으로 파악할 수 있습니다.
            **추세 분석**: 특정 기간 동안 뉴스 발행량이 증가하거나 감소하는 경향을 분석할 수 있습니다.
            **이벤트 연관성**: 뉴스 발행량의 급증이 특정 이벤트나 이슈와 연관되어 있는지 파악할 수 있습니다.
            현재 주어진 데이터로는 어떤 넷플릭스에 어떤 이슈가 발생하였는지 등은 알 수 없지만, 특정 이벤트에 대한 정보를 얻는다면
            본 차트를 통해 일별로 발행량을 정확하고 직관적으로 파악할 수 있습니다.
                """)
# ======================================================
# 11) Altair: 상위 단어의 일별 등장 빈도 추이
# ======================================================
st.header(f"7. 상위 {ts_top_n}개 키워드 일별 등장 추이 (Altair)")
if "news_df" in st.session_state and not st.session_state["news_df"].empty and not time_series_df.empty:

    # Altair 차트 생성
    chart = alt.Chart(time_series_df).mark_line().encode(
        # x축: 날짜 (시계열)
        x=alt.X('날짜:T', title='날짜'),
        # y축: 빈도 (정량적 데이터)
        y=alt.Y('빈도:Q', title='빈도'),
        # 색상: 단어별 구분
        color=alt.Color('단어:N'),
        # 툴팁: 마우스 오버 시 상세 정보 표시
        tooltip=['날짜:T', '단어:N', '빈도:Q']
    ).properties(
        title=f"일별 상위 {ts_top_n}개 키워드 등장 빈도 변화"
    ).interactive() # 줌/패닝 가능하도록 설정
    
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning(f"상위 {ts_top_n}개 키워드 일별 등장 추이 데이터를 찾을 수 없습니다. (데이터 수집 및 분석 확인 필요)")
st.info("""
         **💡 시각화 해석 가이드**
         **선 그래프**: 각 키워드의 일별 등장 빈도 변화를 시각적으로 비교할 수 있습니다.
         **색상**: 각 키워드는 고유한 색상으로 구분되어 있어, 특정 키워드의 추이를 쉽게 파악할 수 있습니다.
         **툴팁**: 마우스 오버 시 해당 날짜와 키워드의 정확한 빈도를 확인할 수 있습니다.
         결론 : 키워드들 모두 동일한 추세로 증가하고 감소하는 폭을 보이고 있다.
         이는 상위 10개 키워드의 추이가 비슷하다, 서로 상관이 있다는 것을 의미할 수 있다.
         """)
