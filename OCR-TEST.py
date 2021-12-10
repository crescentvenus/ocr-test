import streamlit as st
import cv2
from PIL import Image           # 画像処理ライブラリ
import numpy as np              # データ分析用ライブラリ
import unicodedata
import pprint
from typing import List, Set, Tuple
import time


def erase_lines(img,img_thresh,th1):
    # OpenCVで直線の検出
    # https://qiita.com/tifa2chan/items/d2b6c476d9f527785414
    img2 = img.copy()
    img3 = img.copy()
    gray = cv2.cvtColor(img_thresh, cv2.COLOR_BGR2GRAY)
    gray_list = np.array(gray)
    gray2 = cv2.bitwise_not(gray)
    gray2_list = np.array(gray2)
    #lines = cv2.HoughLinesP(gray2, rho=1, theta=np.pi/360, threshold=th1, minLineLength=80, maxLineGap=5)
    lines = cv2.HoughLinesP(gray2, rho=1, theta=np.pi/360, threshold=th1, minLineLength=150, maxLineGap=5)
    xmin,ymin=500,500
    xmax,ymax=0,0
    if lines is not None:

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1<xmin:
                xmin=x1
            if y1<ymin:
                ymin=y1
            if x1>xmax:
                xmax=x1
            if y1>ymax:
                ymax=y1

            # 緑色の線を引く
            red_lines_img = cv2.line(img2, (x1,y1), (x2,y2), (0,255,0), 3)
            red_lines_np=np.array( red_lines_img)

            # 線を消す(白で線を引く)
            no_lines_img = cv2.line(img_thresh, (x1,y1), (x2,y2), (255,255,255), 3)
            no_lines=np.array( no_lines_img)
        dx=int(0.5+(xmax-xmin)/9)
        dy=int(0.5+(ymax-ymin)/9)
        sx=int(0.5+dx*0.05)
        sy=int(0.5+dy*0.05)
        st.write(xmin,ymin,xmax,ymax,dx,dy)
        peaces=[]
        p_size=[]
        p_max=0
        for y in range(9):
            for x in range(9):
                p = xmin + x*dx + sx
                q = ymin + y*dy + sy
                cv2.rectangle(no_lines,(p,q),(p+dx-sx,q+dy-sy),(0,0,255),1)
                # １コマの画像切り出し
                peace=(cv2.cvtColor(no_lines_img[q:q+dy-sy,p:p+dx-sx],cv2.COLOR_BGR2RGB))
                peaces.append(peace)
                # ピクセルの総和の最大値を求める。（数字ナシのコマ）
                ar = np.array(peace).sum()
                if ar>p_max:
                    p_max=ar
                p_size.append(ar)
                #if x == 0:
                #    st.image(peace,caption=str(x)+','+str(y))
        im_h= cv2.hconcat([red_lines_img, no_lines])
    else:
        im_h = None
        no_lines = img_thresh
    return im_h, no_lines,peaces,p_size,p_max

def main():
    st.title('数独の文字認識')
    col1, col2 ,col3 = st.columns([3,1,1])
    KEI = False
    OCR = False
    with col1:
        uploaded_file = st.file_uploader("画像ファイルを選択してアップロード")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = np.array(img)
        th2 = st.slider(label='２値化の閾値',min_value=0, max_value=255, value=100)
        th1 = st.slider(label='線消去の閾値',min_value=0, max_value=255, value=100)
        with col2:
            LNG =  st.selectbox("言語選択",['eng','jpn'])
        with col3:
            KEI = st.checkbox('問題の分割')
        #with col4:
        #    OCR = st.checkbox('OCR実行')

        ret, img_thresh = cv2.threshold(img, th2, 255, cv2.THRESH_BINARY)
        im_h = cv2.hconcat([img, img_thresh])
        st.image(im_h, caption='元画像<--->２値化画像')
        if KEI:
            time_start = time.time()
            im_h, no_lines, peaces,p_size,p_max = erase_lines(img,img_thresh,th1)
            if im_h is None:
                st.warning('No line detectd')
            else:
                new_image = cv2.cvtColor(im_h, cv2.COLOR_BGR2RGB)
                st.image(new_image,caption='線を削除した画像')
                st.write('Time Prep:',time.time() -  time_start)
        else:
            no_lines=img_thresh

        if OCR:
            my_bar = st.progress(0)
            st.subheader('[OCR結果]')
            conf='-l ' + LNG + ' --psm 6  outputbase digits'
            n=0
            row=[]
            #print(p_max,p_size)
            time_start = time.time()
            for peace in peaces:
                ar = np.array(peace)
                t = p_size[n]
                if t<p_max:
                    txt=pytesseract.image_to_string(peace, config=conf)
                    txt=remove_control_characters(txt)
                    ans=int(txt)
                else:
                    ans=0
                row.append(ans)
                my_bar.progress(int(100*n/80))
                n=n+1

            row2=np.array(row).reshape(-1,9).tolist()
            st.success(row2)
            st.write('Time OCR:',time.time() -  time_start)
            grid = solver.Grid(row2)
            results = solver.solve_all(grid)
            st.subheader('[数独回答]')
            m1="<span style=\"color: darkgray; \">"
            m2="　</span>"
            msg="### "
            for r in results:
                buf=[]
                for y in range(9):
                    for x in range(9):
                        buf.append(r._values[y][x])
                        c = r._values[y][x]
                        c = str(c)
                        d = row2[y][x]
                        if d != 0:
                            msg=msg + m1 + c + m2
                        else:
                            msg=msg + c + '　'
                    msg=msg + '<br>'
                msg=msg + "<br>"
                st.markdown(msg,unsafe_allow_html=True)

if __name__ == '__main__':
    main()
