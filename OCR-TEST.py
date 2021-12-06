import streamlit as st
import cv2
from PIL import Image           # 画像処理ライブラリ
import numpy as np              # データ分析用ライブラリ
import pytesseract              # tesseract の python 用ライブラリ

def main():
    st.title('文字認識の実験')
    col1, col2 ,col3 = st.columns([3,1,1])
    OCR = False
    KEI = False
    with col1:
        uploaded_file = st.file_uploader("画像ファイルを選択してアップロード")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = np.array(img)
        th2 = st.slider(label='２値化の閾値',min_value=0, max_value=255, value=100)
        th1 = st.slider(label='線消去の閾値',min_value=0, max_value=255, value=100)
        with col2:
            LNG =  st.selectbox("言語選択",['jpn','eng'])
        with col3:
            KEI = st.checkbox('線削除')
        #with col4:
        #    OCR = st.checkbox('OCR実行')

        ret, img_thresh = cv2.threshold(img, th2, 255, cv2.THRESH_BINARY)
        im_h = cv2.hconcat([img, img_thresh])
        st.image(im_h, caption='元画像<--->２値化画像')
        if KEI:
            img2 = img.copy()
            img3 = img.copy()
            gray = cv2.cvtColor(img_thresh, cv2.COLOR_BGR2GRAY)
            gray_list = np.array(gray)
            #img2.image(gray_list, caption='GRAY',use_column_width=True)
            gray2 = cv2.bitwise_not(gray)
            gray2_list = np.array(gray2)
            lines = cv2.HoughLinesP(gray2, rho=1, theta=np.pi/360, threshold=th1, minLineLength=80, maxLineGap=5)
            if lines is not None:

                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    # 緑色の線を引く
                    red_lines_img = cv2.line(img2, (x1,y1), (x2,y2), (0,255,0), 3)
                    red_lines_np=np.array( red_lines_img)

                    # 線を消す(白で線を引く)
                    no_lines_img = cv2.line(img_thresh, (x1,y1), (x2,y2), (255,255,255), 3)
                    no_lines=np.array( no_lines_img)
                im_h = cv2.hconcat([red_lines_img, no_lines_img])
                st.image(im_h,caption='線を削除した画像')
            else:
                st.warning('No line detectd')
        else:
            no_lines=img_thresh

        if OCR:
            #txt = pytesseract.image_to_string(no_lines, lang="eng",config='--psm 11')
            conf='-l ' + LNG + ' --psm 6'
            txt=pytesseract.image_to_string(no_lines, config=conf)
            st.subheader('---認識結果---')
            st.write(txt)

if __name__ == '__main__':
    main()
