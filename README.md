# pFL_distributed_system
Optimal personalized Federated Learning framework in heterogeneous distributed system

Week 1: Learn the basics of deep learning(Perceptron, MLP, Backpropagation, Gradient Descent) and implement a toy example (MNIST, FashionMNIST, CIFAR-10)

Week 2: Learn Convolutional Neural Networks (CNN) and their applications

Week 3: Learn Federated Learning and simulate FL tutorial on different datasets

Week 4: Learn how to extract personal sleep patterns from multiple devices and store them locally


# pFL_simulation Prerequisites
pip install torch flwr pandas numpy scikit-learn


# One way to make git useful 
1. 작업할 디렉토리(위치)에 새로운 폴더 pFL이름으로 만들기 (위치는.. 404, 바탕화면, etc...)
2. VScode를 열고 Open Folder를 선택해서 방금 만든 pFL 폴더 클릭하면 pFL 폴더 내의 파일들 수정가능
3. VScode에서 Open Terminal 선택해서 현재 위치 확인 (pwd (present working directory): 현재 위치 확인)
4. 현재 위치가 pFL이 아니라면 ls (ls (list): 현재 디렉토리의 하위 파일) 와 cd (cd (change directory){폴더이름}: 폴더이름으로 이동)를 통해서 pFL 폴더로 이동
5. 만약에 pFL 폴더로 이동했다면 현재 빈 폴더일테니 ls하면 아무것도 안나옴 
6. 현재 위치가 pFL이라면, *"git init"* (Initialize empty git repository)을 입력하면 빈 깃 repository 생성됨
7. *"git status"*로 현재 git repository의 상태확인 가능 (예: Modified: CNN.ipynb) 현재로써는 아무것도 없을것임
8. *"git add 추가할 파일이름"*으로 github에 업로드 하고자 하는 파일이나 폴더 선택가능. 보통은 .gitignore에 올리지 않을 데이터 이름 명시하고 git add . (.은 all이라는 뜻)해서 add 함
9. *"git commit -m "업데이트 할 때 기록하는 메세지"*으로 업데이트의 내용을 간략하게 설명해줘야함 (예: 260823 수정내용: CNN.ipynb 코드 수정)
10. *"git push -u origin main"*로 최종 github에 올리는 방식
11. github 웹페이지를 새로고침하면 수정 내용과 commit 메세지 확인 가능
12. 남이 작업한 내용은 자동으로 내 컴퓨터에 업데이트가 되지 않기에 최신 수정본으로 작업을 시작하고 싶으면 *"git pull"*로 현재 제일 최신 내용 가져오기하면 됨
13. 아예 새로운 폴더로 가져오고 싶으면 *"git clone"* 하면 됨
