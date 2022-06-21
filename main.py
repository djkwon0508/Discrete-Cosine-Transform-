import cv2 as cv
import math
import glob
import sys
import os
import numpy as np

# Set your own file path
image_path = r'/Users/kwondongjae/Library/CloudStorage/OneDrive-개인/멀티미디어/HW4/*.jpeg'
imagefile = glob.glob(image_path)


class DCT_LossyCompression:
    def __init__(self, weight=0, height=0):
        self.DCT_basis = []
        self.freq8 = []
        self.freq4 = []
        self.freq2 = []
        self.IDCT8 = []
        self.IDCT4 = []
        self.IDCT2 = []
        self.w = weight
        self.h = height

    def imageSizeCheck(self, image):
        h, w, c = image.shape
        if h % 8 == 0 and w % 8 == 0:
            return True
        else:
            return False

    def img2gray(self, image):
        imageCheck = self.imageSizeCheck(image)
        if not imageCheck:
            print("입력받은 사진의 해상도를 \"640x480\"으로 조정합니다.")
            image = cv.resize(image, dsize=(640, 480), interpolation=cv.INTER_AREA)
            self.w = 640
            self.h = 480
        else:
            print("입력받은 사진의 해상도는 {}x{}입니다.".format(self.w, self.h))
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return grayImage

    def DCT_constant(self, num):
        if num == 0:
            return math.sqrt(2) / 2
        else:
            return 1

    def createDCT_basis(self):
        # DCT_2D = np.zeros((8, 8, 8, 8))
        DCT_2D = [[[[0 for _ in range(8)] for _ in range(8)] for _ in range(8)] for _ in range(8)]

        for u in range(8):
            for v in range(8):
                for i in range(8):
                    for j in range(8):
                        DCT_2D[u][v][i][j] = (self.DCT_constant(u) * self.DCT_constant(v) / 4) \
                                             * (math.cos((2 * i + 1) * u * math.pi / 16)) \
                                             * (math.cos((2 * j + 1) * v * math.pi / 16))
        # [Output] DCT basis : 8 X 8 matrix
        return DCT_2D

    def image2DCT(self, gray):
        freq8_image = [[0 for _ in range(self.w)] for _ in range(self.h)]
        freq4_image = [[0 for _ in range(self.w)] for _ in range(self.h)]
        freq2_image = [[0 for _ in range(self.w)] for _ in range(self.h)]
        # Run DCT
        for h in range(0, self.h, 8):
            for w in range(0, self.w, 8):
                for u in range(8):
                    for v in range(8):
                        for i in range(8):
                            for j in range(8):
                                # Save 8 X 8 Original Freq Matrix
                                freq8_image[h+u][w+v] += self.DCT_basis[u][v][i][j] * gray[h+i][w+j]
                                # [Condition] Save 4 X 4 Freq Matrix
                                if (0 <= u) and (u < 4) and (0 <= v) and (v < 4):
                                    freq4_image[h+u][w+v] += self.DCT_basis[u][v][i][j] * gray[h+i][w+j]
                                else:
                                    freq4_image[h+u][w+v] += 0
                                # [Condition] Save 2 X 2 Freq Matrix
                                if (0 <= u) and (u < 2) and (v <= j) and (v < 2):
                                    freq2_image[h+u][w+v] += self.DCT_basis[u][v][i][j] * gray[h+i][w+j]
                                else:
                                    freq2_image[h+u][w+v] += 0
        return freq8_image, freq4_image, freq2_image

    def DCT2image(self, freq):
        IDCT_2D = [[0 for _ in range(self.w)] for _ in range(self.h)]
        # Calculate 2D IDCT
        for h in range(0, self.h, 8):
            for w in range(0, self.w, 8):
                for i in range(8):
                    for j in range(8):
                        for u in range(8):
                            for v in range(8):
                                IDCT_2D[h+i][w+j] += self.DCT_basis[u][v][i][j] * freq[h+u][w+v]
        return IDCT_2D

    def runDCT_LossyCompression(self, image, DCT_basis):
        # BGR scale -> Gray scale
        gray = self.img2gray(image)
        # Create DCT_basis
        self.DCT_basis = DCT_basis
        # Image -> DCT
        self.freq8, self.freq4, self.freq2 = self.image2DCT(gray)
        # DCT -> image
        image_freq8 = self.DCT2image(self.freq8)
        image_freq4 = self.DCT2image(self.freq4)
        image_freq2 = self.DCT2image(self.freq2)

        return image_freq8, image_freq4, image_freq2

    def print_DCT_basis(self, DCT_basis, v=0):
        self.DCT_basis = list(DCT_basis)
        print("<v={}>일 때 DCT basis matrix는...".format(v))
        for u in range(8):
            print("<u = {}>".format(u))
            for i in range(8):
                print("[", end=' ')
                for j in range(8):
                    print("{:.5f},".format(self.DCT_basis[u][v][i][j]), end=' ')
                print("]")


# Main Function
def main():
    # DCT_basis = [[[[0 for _ in range(8)] for _ in range(8)] for _ in range(8)] for _ in range(8)]
    # Clear Image File
    [os.remove(f) for f in glob.glob("/Users/kwondongjae/PycharmProjects/DCT_lossy_compression/*.png")]
    print("Clear Image Files")
    for imgNum in imagefile:
        if imgNum is None:
            sys.exit("[ERROR] Cannot read Image")
        else:
            print("Processing image file: {}".format(imgNum))
            image = cv.imread(imgNum, cv.IMREAD_COLOR)
            startDCT = DCT_LossyCompression(weight=image.shape[1], height=image.shape[0])
            # Create DCT_basis
            DCT_basis = startDCT.createDCT_basis()
            image_DCT8, image_DCT4, image_DCT2 = startDCT.runDCT_LossyCompression(image, DCT_basis)

            image_IDCT8 = np.array(image_DCT8, dtype=np.float64)
            image_IDCT4 = np.array(image_DCT4, dtype=np.float64)
            image_IDCT2 = np.array(image_DCT2, dtype=np.float64)

            if not ((image_IDCT8 is None) or (image_IDCT4 is None) or (image_IDCT2 is None)):
                print("{}X{} 해상도의 사진을 출력합니다.".format(image_IDCT8.shape[1], image_IDCT8.shape[0]))
                # Image 이름 추출하기
                find_slash = imgNum.rfind('/')
                find_extension = imgNum.rfind('.')
                if find_slash == -1:
                    print("[ERROR] 경로 위치가 없습니다.")
                elif find_extension == -1:
                    print("[ERROR] 파일명에서 확장자를 찾을 수 없습니다.")
                else:
                    original_image_name = imgNum[find_slash+1:find_extension]
                    # Image 저장
                    # cv.imshow("before", image)
                    cv.imwrite(original_image_name + "_DCT8.png", image_IDCT8, [cv.IMWRITE_PNG_COMPRESSION])
                    cv.imwrite(original_image_name + "_DCT4.png", image_IDCT4, [cv.IMWRITE_PNG_COMPRESSION])
                    cv.imwrite(original_image_name + "_DCT2.png", image_IDCT2, [cv.IMWRITE_PNG_COMPRESSION])
                    # 아무 키 입력 시 다음 명령 수행
                    cv.waitKey(0)
                    cv.destroyAllWindows()
            else:
                print("[ERROR] Cannot read IDCT Image.")
    # Print DCT_basis if you want
    # startDCT.print_DCT_basis(DCT_basis, v=0)


# Run or Debug Main function
if __name__ == "__main__":
    main()
