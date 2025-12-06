rl-dqn-cartpole/
├─ src/
│ ├─ train_cartpole.py
│ ├─ dqn_agent.py
│ ├─ replay_buffer.py
│ └─ networks.py
├─ results/
│ ├─ returns_seed0.npy
│ ├─ returns_seed1.npy
│ ├─ returns_seed2.npy
│ ├─ dqn_cartpole_seed0.png
│ ├─ dqn_cartpole_seed1.png
│ └─ dqn_cartpole_seed2.png
├─ requirements.txt
└─ DQN_experiments.ipynb

---

#Conda 환경 생성 및 활성화  
(Anaconda Prompt 실행)

```bash
conda create -n dqn_env python=3.10 -y
conda activate dqn_env

#프로젝트 폴더이동
cd C:\Users\User\rl-dqn-cartpole

#필수 라이브러리 설치
pip install -r requirements.txt

#CartPole 학습 실행
(500 에피소드, seed = 0 예시)
python src/train_cartpole.py --episodes 500 --seed 0 --save-dir results

#여러 random seed로 실험 실행
python src/train_cartpole.py --episodes 500 --seed 0 --save-dir results
python src/train_cartpole.py --episodes 500 --seed 1 --save-dir results
python src/train_cartpole.py --episodes 500 --seed 2 --save-dir results
