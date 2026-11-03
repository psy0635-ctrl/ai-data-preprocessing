"""Create a Korean PDF from measured Iris experiment results only."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "iris_results"
OUT = ROOT / "과제" / "인공지능학습데이터전처리_실습과제3_2601110288_박수용.pdf"
NAMES = {"baseline": "기본 모델", "augmentation": "데이터증강", "dropout": "Dropout", "both": "증강 + Dropout"}


def main():
    d = json.loads((RESULTS / "training_history.json").read_text(encoding="utf-8"))
    pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
    pdfmetrics.registerFont(TTFont("MalgunBold", "C:/Windows/Fonts/malgunbd.ttf"))
    pdfmetrics.registerFontFamily("Malgun", normal="Malgun", bold="MalgunBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="KBody", fontName="Malgun", fontSize=9.2, leading=15, spaceAfter=7, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="KTitle", fontName="MalgunBold", fontSize=20, leading=28, spaceAfter=17))
    styles.add(ParagraphStyle(name="KHeading", fontName="MalgunBold", fontSize=13, leading=20, spaceAfter=9, textColor=colors.HexColor("#24476b")))
    styles.add(ParagraphStyle(name="KSmall", fontName="Malgun", fontSize=7.7, leading=12, spaceAfter=5, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="KCell", fontName="Malgun", fontSize=8, leading=12, alignment=TA_CENTER, wordWrap="CJK"))
    story=[]
    def p(text, style="KBody"):
        story.append(Paragraph(text, styles[style]))
    def heading(text): p(text,"KHeading")
    def image(name, width=495):
        from PIL import Image as PILImage
        path=RESULTS/name
        with PILImage.open(path) as im: height=width*im.height/im.width
        story.append(Image(str(path),width=width,height=height))
        story.append(Spacer(1,8))
    def table(rows,widths):
        t=Table([[Paragraph(escape(str(c)),styles["KCell"]) for c in row] for row in rows],colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e9eff5")),
                              ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                              ("BOTTOMPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),
                              ("LINEBELOW",(0,0),(-1,0),0.8,colors.HexColor("#24476b")),
                              ("LINEBELOW",(0,1),(-1,-1),0.3,colors.HexColor("#d7dde3"))]))
        story.append(t);story.append(Spacer(1,10))
    def page(): story.append(PageBreak())
    def footer(canvas,doc):
        canvas.setFont("Malgun",8)
        canvas.setFillColor(colors.HexColor("#66717c"))
        canvas.drawString(42,27,"인공지능학습데이터전처리 | 실습과제 3 | 2601110288 박수용")
        canvas.drawRightString(A4[0]-42,27,str(doc.page))

    p("인공지능학습데이터 전처리<br/>실습과제 3", "KTitle")
    p("Iris 데이터의 데이터증강 및 Dropout 적용 결과 비교", "KHeading")
    p("학번: 2601110288　 이름: 박수용　 작성일: 2026년 11월 3일")
    heading("1. 실험 목적과 데이터 확인")
    p("Iris 데이터를 이용하여 기본 모델, 데이터증강 적용 모델, Dropout 적용 모델, 두 기법을 함께 적용한 모델의 Training / Validation Accuracy와 Loss를 비교한다.")
    p("제공된 Flower.py와 Flower2.py는 꽃 사진 5종을 분류하는 CNN이다. 과제에 명시된 Iris와 다른 데이터이므로 그대로 사용하지 않았다. 프로젝트에서 기존 Iris 모델을 찾지 못해 Iris용 기본 신경망을 새로 정의하고, Flower2.py의 네 조건 비교 방식을 참고하였다.")
    p("scikit-learn의 load_iris를 사용하였다. 표본은 150개이며, 입력은 꽃받침 길이·너비와 꽃잎 길이·너비의 4개 수치 특성이다. 클래스는 setosa, versicolor, virginica의 3종이며 각각 50개이다.")
    heading("2. 공통 학습 조건")
    table([["항목","설정"],["데이터 분할","계층화 분할: 학습 120개(클래스별 40), 검증 30개(클래스별 10)"],
           ["전처리","학습 데이터로만 StandardScaler 적합, 동일 변환을 검증 데이터에 적용"],
           ["기본 모델","입력 4 → Dense(32, ReLU) → Dense(16, ReLU) → Dense(3, Softmax)"],
           ["학습 설정",f"Adam, 학습률 0.001, batch 16, {d['epochs']} epochs, 교차엔트로피"],
           ["반복 시드",", ".join(map(str,d["seeds"]))]], [92,403])
    p("시드별로 네 모델에 동일한 분할·초기 Dense 가중치·배치 순서를 적용하였다. 별도 테스트 데이터와 검증 결과를 이용한 하이퍼파라미터 탐색은 사용하지 않았다. 결과표는 모두 동일한 마지막 epoch 기준이다.","KSmall")
    page()

    heading("3. 네 가지 비교 조건")
    table([["모델","데이터증강","Dropout"],["기본 모델","없음","없음"],
           ["데이터증강","GaussianNoise(0.1)","없음"],["Dropout","없음","각 은닉층 뒤 0.2"],
           ["증강 + Dropout","GaussianNoise(0.1)","각 은닉층 뒤 0.2"]],[125,185,185])
    p("Iris는 이미지가 아닌 수치 데이터이므로 회전·뒤집기 대신 표준화한 입력에 평균 0, 표준편차 0.1의 가우시안 잡음을 더하였다. 증강은 입력층 다음에 위치하며 학습할 때만 작동한다. 저장된 표본 수를 늘리는 방식이 아니라 매 학습 배치에 변형을 적용하는 온라인 증강이다.")
    p("Dropout은 각 은닉층의 출력 일부를 학습 중 20% 확률로 제외한다. 검증할 때는 잡음과 Dropout을 모두 비활성화하여 원본 검증 자료에 대한 성능을 측정한다.")
    heading(f"4. 네 조건의 학습 곡선 (대표 시드 {d['seeds'][0]})")
    image("comparison_primary.png")
    p("가로축은 epoch, 세로축은 정확도 또는 손실이다. 네 모델 모두 같은 축 범위와 학습 횟수로 비교하였다. 대표 시드는 결과를 보기 전에 정한 첫 번째 시드이며, 가장 좋은 결과를 고른 것이 아니다.","KSmall")
    page()

    heading("5. 모델별 Training / Validation 곡선")
    image("history_baseline.png")
    image("history_augmentation.png")
    p("Training 값은 학습 중 기록된 값이다. 데이터증강·Dropout 모델에서는 학습 시 변형과 무작위 제외가 적용되므로 Validation 값과 측정 조건이 다르다. Validation Accuracy가 Training Accuracy보다 높다는 이유만으로 오류라고 판단하지 않는다.","KSmall")
    page()
    heading("5. 모델별 Training / Validation 곡선 (계속)")
    image("history_dropout.png")
    image("history_both.png")
    p("Accuracy는 맞힌 비율이고 Loss는 정답 클래스 확률을 반영한 교차엔트로피이다. 정확도가 같아도 예측 확률의 확신 정도가 다르면 손실이 달라질 수 있다.","KSmall")
    page()

    heading(f"6. 마지막 epoch 결과 (대표 시드 {d['seeds'][0]})")
    primary=d['runs'][0]
    rows=[["모델","Train Acc","Val Acc","Train Loss","Val Loss"]]
    for name in NAMES:
        h=primary['histories'][name]
        rows.append([NAMES[name],f"{h['accuracy'][-1]*100:.2f}%",f"{h['val_accuracy'][-1]*100:.2f}%",f"{h['loss'][-1]:.4f}",f"{h['val_loss'][-1]:.4f}"])
    table(rows,[115,95,95,95,95])
    heading(f"7. 반복 실험 결과 ({len(d['seeds'])}개 시드)")
    rows=[["모델","Val Accuracy\n평균 ± 표본 SD","Val Loss\n평균 ± 표본 SD"]]
    for name in NAMES:
        a=d['summary'][name]['val_accuracy']; l=d['summary'][name]['val_loss']
        rows.append([NAMES[name],f"{a['mean']*100:.2f}% ± {a['sample_sd']*100:.2f}%p",f"{l['mean']:.4f} ± {l['sample_sd']:.4f}"])
    table(rows,[135,180,180])
    image("comparison_mean.png",width=460)
    p("실선은 시드별 곡선의 평균, 음영은 표본 표준편차(±1 SD)이다. 반복마다 데이터 분할과 초기화가 함께 바뀌므로 두 변동 요인이 포함된다. 음영은 신뢰구간이 아니다.","KSmall")
    page()

    heading("8. 결과 해석")
    base=d['summary']['baseline']
    for name in NAMES:
        s=d['summary'][name]; acc=s['val_accuracy']['mean']; loss=s['val_loss']['mean']
        delta=(acc-base['val_accuracy']['mean'])*100
        dl=loss-base['val_loss']['mean']
        if name=='baseline':
            p(f"<b>기본 모델:</b> 반복 실험의 평균 검증 정확도는 {acc*100:.2f}%, 평균 검증 손실은 {loss:.4f}로 나타났다. 다른 조건을 비교하는 기준으로 사용하였다.")
        else:
            p(f"<b>{NAMES[name]}:</b> 평균 검증 정확도는 {acc*100:.2f}%, 평균 검증 손실은 {loss:.4f}였다. 기본 모델 대비 정확도 차이는 {delta:+.2f}%p, 손실 차이는 {dl:+.4f}이다. 정확도는 높을수록, 손실은 낮을수록 유리하다.")
    best_acc=max(d['summary'],key=lambda n:d['summary'][n]['val_accuracy']['mean'])
    best_loss=min(d['summary'],key=lambda n:d['summary'][n]['val_loss']['mean'])
    p(f"이 설정에서 평균 검증 정확도가 가장 높은 조건은 {NAMES[best_acc]}, 평균 검증 손실이 가장 낮은 조건은 {NAMES[best_loss]}이었다. 이는 본 실험의 기술적 비교이며 통계적 유의성이나 보편적 우월성을 의미하지 않는다. 동일 평균이 있으면 조건 정의 순서의 첫 항목을 표시하였다.")
    heading("9. 해석 범위와 결론")
    p("Iris는 작은 데이터셋이며 검증 표본은 30개이다. 검증 표본 한 개의 정답 여부가 정확도를 약 3.33%p 바꾸므로, 단일 실행의 작은 차이를 크게 해석하기 어렵다. 서로 다른 시드의 분할은 표본이 겹칠 수 있어 독립적인 외부 평가로 볼 수 없다.")
    p("학습 중 기록한 Training 지표는 정규화 기법이 활성화된 상태의 값이다. 이를 보완하기 위해 잡음과 Dropout을 끈 상태의 학습 데이터 재평가 값도 CSV와 JSON에 별도로 저장하였다. 모델 비교는 동일한 원본 검증 데이터의 지표를 중심으로 수행하였다.")
    p("수치형 잡음 증강은 측정값의 작은 변동을 가정한 실험적 방법이다. 실제 식물 측정의 생물학적 타당성을 보장하지 않으며, 잡음 크기·Dropout 비율·모델 크기가 달라지면 결과도 달라질 수 있다. 두 기법을 함께 사용한다고 항상 성능이 향상되는 것은 아니다.")
    p("네 조건의 학습·검증 Accuracy 및 Loss를 비교하여 데이터증강과 Dropout의 효과를 확인하였다. 이번 실험에서는 실제 검증 결과와 반복 변동을 함께 제시하였으며, 최고 학습 정확도만으로 모델의 일반화 성능을 판단하지 않았다.")
    page()

    heading("10. 재현 방법과 검증")
    p("실행 파일: Iris_assignment3.py<br/>보고서 생성: make_iris_report.py<br/>결과 폴더: iris_results<br/>패키지 버전: requirements-iris.lock.txt")
    p("실행 순서: 전용 가상환경의 Python으로 Iris_assignment3.py를 실행한 뒤 make_iris_report.py를 실행한다. 학습 코드는 외부 꽃 이미지 다운로드가 필요 없으며 scikit-learn에 포함된 Iris를 사용한다.")
    p("검증 항목: 학습·검증 인덱스 중복 없음, 클래스별 분할 수 확인, 학습 자료만 사용한 스케일러, 네 모델의 동일 초기 Dense 가중치, 추론 시 증강·Dropout 비활성화, 모든 epoch 지표의 유한값, 마지막 검증 지표와 재평가 결과의 일치.")
    p("실행 환경: " + ", ".join(f"{escape(k)} {escape(v)}" for k,v in d['versions'].items()),"KSmall")
    heading("참고 자료")
    refs=[("Iris 데이터 구성", "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html"),
          ("데이터 누수와 전처리", "https://scikit-learn.org/stable/common_pitfalls.html"),
          ("GaussianNoise: 수치형 입력의 학습 시 잡음 증강", "https://keras.io/api/layers/regularization_layers/gaussian_noise/"),
          ("Dropout: 학습 시 활성화, 추론 시 비활성화", "https://keras.io/api/layers/regularization_layers/dropout/"),
          ("원본 Flower 코드의 꽃 이미지 분류 참고", "https://www.tensorflow.org/tutorials/images/classification")]
    for title,url in refs:
        p(f'{escape(title)}<br/><link href="{url}" color="#24476b">{url}</link>',"KSmall")
    p("원본 과제의 Iris 사용 요구사항을 기준으로 작성하였다. Flower.py·Flower2.py의 CNN을 그대로 사용한 실험 결과가 아니며, 프로젝트에 기존 Iris 모델이 없어 새로 정의한 MLP를 베이스모델로 사용하였다.","KSmall")
    OUT.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=44,
                      title="인공지능학습데이터전처리 실습과제 3",author="박수용").build(story,onFirstPage=footer,onLaterPages=footer)
    print(OUT)


if __name__=='__main__':
    revised_builder = ROOT / 'form_revision' / 'build_form.py'
    if revised_builder.exists():
        import runpy
        runpy.run_path(str(revised_builder), run_name='__main__')
    else:
        main()
