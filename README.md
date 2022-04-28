# Recommender-System-Project
영화 추천을 위한 추천 모델링 프로젝트

## 프로젝트 설명
  + 영화평점 데이터를 이용한 추천시스템 구현
  + 데이터를 입력받고 User id를 index, 영화 id를 column으로 하고 평점을 row로 하는 sparse matrix 빌드
  + Matrix Factorization을 진행하여 sparse matrix를 예측 평점으로 채우기
  + Test data를 입력받아 추천할 영화의 평점을 예측
## Tech & Skill
  + python, PureSVD 알고리즘을 이용한 classification, 편미분
## 추천 알고리즘 구현 결과
<img width="796" alt="_2021-06-06__1 22 09" src="https://user-images.githubusercontent.com/83147205/165778383-4c452a87-a4ea-4f71-b596-2566b9c6c1a0.png">
0으로 되어 있는 부분은 평점이 없는 영화들이다.
구현한 모델을 적용하여 0으로 되어 있던 부분에 예측된 평점으로 채운다.

## 실행 방법
  + python.py recommender u1.base u1.test
## input data
  + 아이디, 영화제목, 평점, timestamp로 구성된 데이터가 train data로 입력된다.
  + 영화제목을 숫자로 바꾸고 모델링 시작
