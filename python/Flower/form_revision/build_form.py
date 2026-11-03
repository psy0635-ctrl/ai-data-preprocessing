"""Fill the actual assignment HWPX form; export natively via Hancom."""
import copy
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
D=json.loads((BASE/'iris_results/training_history.json').read_text(encoding='utf-8'))
root=ET.parse(HERE/'template_native.hml').getroot()
seed=ET.parse(HERE/'picture_seed.hml').getroot()
section=root.find('.//SECTION')
first,continuation=[copy.deepcopy(x) for x in section]
for x in list(section):section.remove(x)
chars=root.find('.//CHARSHAPELIST');paras=root.find('.//PARASHAPELIST')

def char_style(height,bold=False):
    x=copy.deepcopy(chars[6]);n=len(chars);x.set('Id',str(n));x.set('Height',str(height))
    if bold and x.find('BOLD') is None:ET.SubElement(x,'BOLD')
    chars.append(x);chars.set('Count',str(len(chars)));return str(n)

def para_style(align='Justify',line=155,before=0,after=230,keep=False):
    x=copy.deepcopy(paras[0]);n=len(paras);x.set('Id',str(n));x.set('Align',align);x.set('KeepWithNext',str(keep).lower())
    x.find('PARAMARGIN').attrib.update({'LineSpacing':str(line),'Prev':str(before),'Next':str(after)})
    paras.append(x);paras.set('Count',str(len(paras)));return str(n)

CS={'body':char_style(1000),'title':char_style(1200,True),'head':char_style(1100,True),
    'caption':char_style(900),'table':char_style(900),'small':char_style(850)}
PS={'body':para_style(),'title':para_style('Center',150,100,400,True),
    'head':para_style('Left',145,350,200,True),'caption':para_style('Center',130,100,300),
    'table':para_style('Center',130,0,0),'figure':para_style('Center',100,0,0),
    'small':para_style('Left',140,0,180)}
mapping=root.find('.//MAPPINGTABLE')
old=mapping.find('BINDATALIST')
if old is not None:mapping.remove(old)
mapping.insert(0,copy.deepcopy(seed.find('.//BINDATALIST')))
tail=root.find('TAIL')
for x in list(tail):
    if x.tag=='BINDATASTORAGE':tail.remove(x)
tail.insert(0,copy.deepcopy(seed.find('.//BINDATASTORAGE')))
pictures=[copy.deepcopy(x) for x in seed.findall('.//PICTURE')]
assert len(pictures)==6
instance=1800000000
def uid():
    global instance
    instance+=1;return str(instance)
def p(text='',kind='body'):
    x=ET.Element('P',{'ParaShape':PS[kind],'Style':'0'})
    run=ET.SubElement(x,'TEXT',{'CharShape':CS.get(kind,CS['body'])})
    if text:ET.SubElement(run,'CHAR').text=text
    return x
def add(c,text,kind='body'):c.append(p(text,kind))
def replace_cell(cell,text):
    listing=cell.find('PARALIST');original=listing.find('P')
    new=ET.Element('P',dict(original.attrib));run=ET.SubElement(new,'TEXT',dict(original.find('TEXT').attrib));ET.SubElement(run,'CHAR').text=text
    for x in list(listing):listing.remove(x)
    listing.append(new)
def page(number):
    node=copy.deepcopy(first if number==1 else continuation)
    node.attrib.pop('InstId',None);node.set('PageBreak','false' if number==1 else 'true')
    table=node.find('.//TABLE');table.find('SHAPEOBJECT').set('InstId',uid());table.find('SHAPEOBJECT').set('ZOrder',str(number-1));table.set('RepeatHeader','false')
    identity=table.findall('ROW')[1].findall('CELL')
    for index,text in [(1,'3'),(3,'2601110288'),(5,'박수용')]:replace_cell(identity[index],text)
    cell=table.findall('ROW')[-1].find('CELL');cell.set('Height','43110' if number==1 else '58500');cell.set('HasMargin','true')
    cell.find('CELLMARGIN').attrib.update({'Left':'680','Right':'680','Top':'400','Bottom':'350'})
    content=cell.find('PARALIST');content.set('VertAlign','Top')
    for x in list(content):content.remove(x)
    if number!=1:table.find('.//SHAPEOBJECT/SIZE').set('Height',str(58500+3*2524))
    section.append(node);return content
def grid(c,headers,rows,widths):
    para=p('','figure');table=ET.SubElement(para.find('TEXT'),'TABLE',{'BorderFill':'3','CellSpacing':'0','ColCount':str(len(headers)),'RowCount':str(len(rows)+1),'PageBreak':'Cell','RepeatHeader':'true'})
    shape=ET.SubElement(table,'SHAPEOBJECT',{'InstId':uid(),'Lock':'false','NumberingType':'Table','ZOrder':'0'})
    ET.SubElement(shape,'SIZE',{'Height':str(1900*(len(rows)+1)),'Width':str(sum(widths)),'HeightRelTo':'Absolute','WidthRelTo':'Absolute','Protect':'false'})
    ET.SubElement(shape,'POSITION',{'AffectLSpacing':'false','AllowOverlap':'false','FlowWithText':'true','HoldAnchorAndSO':'false','HorzAlign':'Center','HorzOffset':'0','HorzRelTo':'Para','TreatAsChar':'true','VertAlign':'Top','VertOffset':'0','VertRelTo':'Para'})
    ET.SubElement(shape,'OUTSIDEMARGIN',{'Bottom':'200','Top':'150','Left':'0','Right':'0'})
    ET.SubElement(table,'INSIDEMARGIN',{'Bottom':'180','Top':'180','Left':'200','Right':'200'})
    for rid,row in enumerate([headers]+rows):
        r=ET.SubElement(table,'ROW')
        for cid,value in enumerate(row):
            cell=ET.SubElement(r,'CELL',{'BorderFill':'3','ColAddr':str(cid),'ColSpan':'1','Dirty':'false','Editable':'false','HasMargin':'true','Header':str(rid==0).lower(),'Height':'1900','Protect':'false','RowAddr':str(rid),'RowSpan':'1','Width':str(widths[cid])})
            ET.SubElement(cell,'CELLMARGIN',{'Bottom':'180','Top':'180','Left':'200','Right':'200'})
            listing=ET.SubElement(cell,'PARALIST',{'LineWrap':'Break','LinkListID':'0','LinkListIDNext':'0','TextDirection':'0','VertAlign':'Center'})
            listing.append(p(str(value),'table'))
    c.append(para)
def picture(c,index,caption):
    pic=copy.deepcopy(pictures[index]);pic.find('SHAPEOBJECT').set('InstId',uid());pic.find('SHAPECOMPONENT').set('InstID',uid())
    pic.find('SHAPEOBJECT/SHAPECOMMENT').text=caption
    para=p('','figure');para.find('TEXT').append(pic);c.append(para);add(c,caption,'caption')

c=page(1)
add(c,'Iris 데이터의 증강 및 Dropout 적용 결과 비교','title')
add(c,'1. 실험 목적과 데이터 전처리','head')
add(c,'이번 과제에서는 Iris 데이터를 사용해 기본 모델과 데이터증강, Dropout, 두 방법을 함께 적용한 모델을 비교했다. 학습 정확도만 높아지는지, 검증 데이터에서도 성능이 좋아지는지 Accuracy와 Loss 그래프로 확인했다.')
add(c,'Iris는 꽃받침 길이와 너비, 꽃잎 길이와 너비를 입력으로 사용하는 데이터이다. 표본은 총 150개이며 setosa, versicolor, virginica가 각각 50개씩 있다. 학습 자료는 120개, 검증 자료는 30개로 나누었다. 각 클래스가 학습에 40개, 검증에 10개씩 들어가도록 비율을 맞췄다.')
add(c,'특성마다 값의 범위가 달라 StandardScaler로 표준화했다. 먼저 데이터를 나눈 다음 학습 자료에서만 평균과 표준편차를 구했다. 검증 자료에는 같은 기준을 적용해 검증 정보가 전처리에 미리 반영되지 않도록 했다.')
add(c,'2. 기본 모델과 공통 학습 조건','head')
add(c,'기존 Flower 코드는 꽃 사진을 분류하는 모델이어서 과제의 Iris 데이터에 그대로 적용하지 않았다. 별도의 Iris 모델이 없어 입력 4개, 은닉층 32개와 16개, 출력 3개인 기본 신경망을 구성했다. 은닉층에는 ReLU, 출력층에는 Softmax를 사용했다.')
add(c,'네 모델 모두 Adam, 학습률 0.001, 배치 크기 16으로 100 epoch를 학습했다. 손실 함수는 sparse categorical crossentropy를 사용했다. 같은 실행 안에서는 데이터 분할, 초기 Dense 가중치, 배치 순서를 맞추고 증강과 Dropout 적용 여부만 바꾸었다.')
add(c,'대표 결과는 시드 123으로 정했다. 한 번의 결과에만 의존하지 않도록 시드 123, 456, 789, 2026, 42로 반복했으며, 모든 결과는 마지막 100번째 epoch를 기준으로 비교했다.')

c=page(2)
add(c,'3. 데이터증강과 Dropout 적용 방법','head')
add(c,'Iris는 수치형 데이터이므로 이미지처럼 회전하거나 뒤집는 대신 입력에 작은 잡음을 더하는 방법을 선택했다. 표준화한 입력에 평균 0, 표준편차 0.1의 GaussianNoise를 적용했다. 학습 중에만 입력값을 조금씩 바꾸는 방식이며, 원본 표본 수 120개가 늘어나는 것은 아니다.')
add(c,'Dropout은 두 은닉층 뒤에 각각 0.2로 적용했다. 학습할 때 출력의 일부를 무작위로 제외해 특정 노드에만 의존하지 않도록 했다. 검증에서는 잡음과 Dropout을 모두 꺼서 원본 검증 자료의 성능을 비교했다.')
grid(c,['비교 모델','데이터증강','Dropout'],[['기본 모델','없음','없음'],['데이터증강','GaussianNoise(0.1)','없음'],['Dropout','없음','은닉층 뒤 각각 0.2'],['증강 + Dropout','GaussianNoise(0.1)','은닉층 뒤 각각 0.2']],[13000,16000,16000])
add(c,'4. 기본 모델과 데이터증강 모델의 학습 결과','head')
picture(c,0,'그림 1. 기본 모델의 Training / Validation Accuracy와 Loss (시드 123)')
add(c,'기본 모델은 학습이 진행되면서 손실이 줄고 정확도가 높아졌다. 마지막 학습 정확도는 98.33%였지만 검증 정확도는 90.00%로 차이가 남았다.')
picture(c,1,'그림 2. 데이터증강 모델의 Training / Validation Accuracy와 Loss (시드 123)')
add(c,'데이터증강 모델도 마지막 검증 정확도는 90.00%였다. 다만 검증 손실은 기본 모델의 0.1482보다 낮은 0.1355로 나타났다. 이 실행에서는 정답 개수는 같았지만 예측 확률을 반영하는 손실에서 차이가 있었다.')

c=page(3)
add(c,'5. Dropout 모델과 두 방법을 함께 적용한 결과','head')
picture(c,2,'그림 3. Dropout 모델의 Training / Validation Accuracy와 Loss (시드 123)')
add(c,'Dropout 모델의 마지막 학습 정확도는 98.33%, 검증 정확도는 90.00%였다. 학습 손실은 0.0869로 기본 모델보다 높았고, 검증 손실은 0.1442로 약간 낮았다. 이 실행만으로 Dropout의 개선 효과가 크다고 보기는 어려웠다.')
picture(c,3,'그림 4. 증강 + Dropout 모델의 Training / Validation Accuracy와 Loss (시드 123)')
add(c,'두 방법을 함께 적용한 모델은 마지막 학습 정확도 96.67%, 검증 정확도 90.00%, 검증 손실 0.1388이었다. 두 방법을 함께 사용했지만 증강만 적용한 모델보다 검증 손실이 더 낮아지지는 않았다.')
add(c,'그래프의 Training 값은 잡음이나 Dropout이 적용된 학습 과정에서 기록된 값이다. 반면 Validation 값은 두 기법을 끈 상태에서 계산된다. 따라서 학습 지표의 차이만으로 어느 모델이 더 좋다고 판단하지 않고 검증 결과를 함께 확인했다.')

c=page(4)
add(c,'6. 같은 조건에서 네 모델 비교','head')
picture(c,4,'그림 5. 네 모델의 학습·검증 정확도 및 손실 비교 (시드 123)')
add(c,'각 그래프에서 같은 지표의 네 곡선을 겹쳐 비교했다. 네 모델 모두 초반에 빠르게 학습했고, 후반에는 검증 정확도가 비슷한 수준에 머물렀다. 마지막 epoch의 수치는 다음과 같다.')
rows=[]
for name,label in [('baseline','기본'),('augmentation','증강'),('dropout','Dropout'),('both','증강+Dropout')]:
    h=D['runs'][0]['histories'][name]
    rows.append([label,f"{100*h['accuracy'][-1]:.2f}%",f"{100*h['val_accuracy'][-1]:.2f}%",f"{h['loss'][-1]:.4f}",f"{h['val_loss'][-1]:.4f}"])
grid(c,['모델','학습 정확도','검증 정확도','학습 손실','검증 손실'],rows,[10000,9000,9000,8500,8500])
add(c,'대표 실행에서는 네 모델이 검증 자료 30개 중 27개를 맞혀 정확도가 모두 90.00%였다. 따라서 이 표에서는 검증 정확도만으로 순위를 나눌 수 없다. 검증 손실은 증강, 증강+Dropout, Dropout, 기본 모델 순으로 낮았다.')
add(c,'학습 정확도는 기본 모델이 높았지만, 검증 손실은 증강 모델이 더 낮았다. 학습 자료를 잘 맞히는 것과 새로운 자료에서 좋은 결과를 내는 것은 구분해서 봐야 한다고 생각했다.')

c=page(5)
add(c,'7. 시드를 바꾼 반복 실험 결과','head')
add(c,'Iris는 자료가 적어 분할에 따라 결과가 달라질 수 있다. 이를 확인하기 위해 시드 5개로 같은 실험을 반복했다. 아래 표는 각 실행의 마지막 epoch 검증 결과를 평균한 값이다. ± 뒤에는 실행 간 차이를 나타내는 표본 표준편차를 표시했다.')
rows=[]
for name,label in [('baseline','기본'),('augmentation','증강'),('dropout','Dropout'),('both','증강+Dropout')]:
    a=D['summary'][name]['val_accuracy'];l=D['summary'][name]['val_loss']
    rows.append([label,f"{100*a['mean']:.2f}% ± {100*a['sample_sd']:.2f}%p",f"{l['mean']:.4f} ± {l['sample_sd']:.4f}"])
grid(c,['모델','평균 검증 정확도 ± 표준편차','평균 검증 손실 ± 표준편차'],rows,[10000,17500,17500])
picture(c,5,'그림 6. 시드 5개에서 얻은 학습 곡선의 평균과 표본 표준편차')
add(c,'증강만 적용했을 때 평균 검증 정확도가 93.33%로 기본 모델보다 1.33%p 높았고, 평균 검증 손실도 0.1229로 가장 낮았다. 기본 모델의 평균 검증 손실은 0.1315였다.')
add(c,'Dropout과 두 방법을 함께 적용한 모델의 평균 검증 정확도는 모두 92.00%로 기본 모델과 같았다. 평균 검증 손실은 각각 0.1544, 0.1462로 기본 모델보다 높았다. 이번 설정에서는 증강만 적용한 경우가 평균적으로 가장 좋은 결과를 보였다.')
add(c,'실선은 5회 평균이며 음영은 평균 ± 표본 표준편차이다. 시드를 바꿀 때 데이터 분할과 초기 가중치가 함께 달라졌다. 음영은 결과의 변동을 보여주는 것으로, 통계적 유의성을 판단하는 신뢰구간은 아니다.','small')

c=page(6)
add(c,'8. 결과 해석과 결론','head')
add(c,'이번 실험에서는 증강만 적용한 모델이 평균 검증 정확도와 손실에서 가장 좋은 결과를 보였다. 입력값을 조금씩 바꿔 학습한 것이 검증 성능에 도움이 되었을 가능성이 있다. 다만 대표 시드에서는 네 모델의 정확도가 같았고, 다른 시드에서 얻은 차이도 크지는 않았다.')
add(c,'Dropout은 과적합을 줄이기 위한 방법이지만, 이번 실험에서 항상 성능을 높여 주지는 않았다. 두 방법을 함께 적용한 모델도 증강만 사용한 모델보다 좋은 결과를 내지 못했다. 사용한 기법의 수보다 데이터와 모델에 맞는 설정이 중요하다는 점을 확인했다.')
add(c,'검증 자료는 30개라서 한 개를 더 맞히거나 틀리면 정확도가 약 3.33%p 달라진다. 따라서 평균 정확도 1.33%p 차이만으로 증강이 언제나 더 좋다고 결론 내릴 수는 없다. 반복 분할의 검증 자료는 서로 겹칠 수 있으며, 별도의 시험 데이터에 대한 성능을 측정한 것도 아니다.')
add(c,'GaussianNoise는 측정값의 작은 변화가 있어도 같은 클래스로 분류하도록 가정한 증강 방법이다. 실제 식물의 측정 오차를 확인해 정한 값은 아니므로 잡음의 크기나 모델 구조를 바꾸면 결과가 달라질 수 있다.')
add(c,'과제를 진행하면서 학습 정확도만 보면 모델을 잘못 판단할 수 있다는 점을 알 수 있었다. 앞으로도 전처리를 학습 자료에만 맞췄는지 먼저 확인하고, 같은 조건에서 검증 정확도와 손실을 함께 비교해야겠다고 생각했다.')
add(c,'9. 사용 도구와 참고 자료','head')
add(c,'실험은 Python 3.11.15와 TensorFlow 2.21.0으로 진행했다. Iris 데이터는 scikit-learn의 load_iris를 사용했다. 전체 20회 학습 기록을 저장했으며, 시드 123의 네 모델을 다시 실행했을 때 원래 학습 기록과 일치했다.','small')
for title,url in [('Iris 데이터 구성','https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html'),('데이터 누수와 전처리','https://scikit-learn.org/stable/common_pitfalls.html'),('GaussianNoise 적용 방법','https://keras.io/api/layers/regularization_layers/gaussian_noise/'),('Dropout 적용 방법','https://keras.io/api/layers/regularization_layers/dropout/')]:
    add(c,title,'small')
    add(c,url,'small')

summary=root.find('.//DOCSUMMARY')
if summary is not None:
    for x in summary:
        if x.tag in ('AUTHOR','CREATOR'):x.text='박수용'
        elif x.tag=='TITLE':x.text='Iris 데이터의 증강 및 Dropout 적용 결과 비교'
ET.ElementTree(root).write(HERE/'report_filled.hml',encoding='utf-8',xml_declaration=True)
text='\n'.join(x.text or '' for x in root.findall('.//CHAR'))
(HERE/'report_plaintext.txt').write_text(text,encoding='utf-8')
assert len(section)==6 and len(root.findall('.//PICTURE'))==6
assert '가스환경' not in text and '과제번호' in text
print('FORM BUILT: 6 pages, 6 plots')

def export():
    import win32com.client
    out=BASE/'과제'; stem='인공지능학습데이터전처리_실습과제3_2601110288_박수용'
    backup=HERE/'backup';backup.mkdir(exist_ok=True)
    for ext in ['.hwpx','.pdf']:
        src=out/(stem+ext);dst=backup/(stem+ext)
        if src.exists() and not dst.exists():shutil.copy2(src,dst)
    h=win32com.client.DispatchEx('HWPFrame.HwpObject')
    try:
        assert h.Open(str(HERE/'report_filled.hml'),'HWPML2X','')
        print('NATIVE_PAGES',h.PageCount,flush=True)
        assert h.SaveAs(str(out/(stem+'.hwpx')),'HWPX','')
        assert h.SaveAs(str(out/(stem+'.pdf')),'PDF','')
        print('SAVED HWPX AND PDF',flush=True)
    finally:h.Quit()

if __name__=='__main__':export()
