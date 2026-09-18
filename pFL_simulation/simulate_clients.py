
import os
import sys
import time
import argparse
import subprocess
import signal

def parse_args():
    parser = argparse.ArgumentParser(description="PFL 다중 클라이언트 시뮬레이션 러너")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pia", "clia"],
        default="pia",
        help="연합학습 모드 선택: 'pia' (가중치 상호작용) 또는 'clia' (로짓 상호작용) (기본값: pia)"
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="연합학습 서버 주소 (기본값: pia는 127.0.0.1:8080, clia는 127.0.0.1:8081)"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=6,
        help="시뮬레이션할 클라이언트 수 (기본값: 6)"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=4,
        help="시뮬레이션 클라이언트 시작 ID (기본값: 4 -> Client #4 ~ #9)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../Donghoon/health_data.csv",
        help="클라이언트들에게 분할 제공할 오픈 데이터셋 경로 (기본값: ../Donghoon/health_data.csv)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 서버 주소 기본값 설정
    if args.server is None:
        args.server = "127.0.0.1:8080" if args.mode == "pia" else "127.0.0.1:8081"
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_script = "client.py" if args.mode == "pia" else "client_CLIA.py"
    target_script_path = os.path.join(script_dir, target_script)
    
    # 데이터 경로 검증 (상대 경로 및 절대 경로 대응)
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.normpath(os.path.join(script_dir, data_path))
        
    if not os.path.exists(data_path):
        print(f"[Error] 데이터셋 파일을 찾을 수 없습니다: {data_path}")
        sys.exit(1)

    print("=" * 70)
    print(f" [PFL Desktop Multi-Client Simulator]")
    print(f" ---------------------------------------------------------------------")
    print(f" ▶ 연합학습 모드       : {args.mode.upper()} ({'파라미터 교환' if args.mode == 'pia' else '로짓 교환'})")
    print(f" ▶ 대상 서버 주소     : {args.server}")
    print(f" ▶ 실행 클라이언트 수 : {args.num_clients}대 (ID #{args.start_id} ~ #{args.start_id + args.num_clients - 1})")
    print(f" ▶ 데이터셋 분할 소스 : {data_path}")
    print(f" ▶ 실행 스크립트       : {target_script}")
    print("=" * 70)
    print(" >> 클라이언트 프로세스들을 시작합니다...")

    processes = []
    
    def cleanup(sig=None, frame=None):
        print("\n\n[Simulator] 중단 요청 감지. 모든 시뮬레이션 클라이언트 프로세스를 종료합니다...")
        for p in processes:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        print("[Simulator] 모든 하위 프로세스가 종료되었습니다.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        for i in range(args.num_clients):
            client_id = args.start_id + i
            # python client.py <server> <client_id> <data_path> <total_partitions>
            cmd = [
                sys.executable,
                target_script_path,
                args.server,
                str(client_id),
                data_path,
                str(args.num_clients)
            ]
            
            # 프로세스 실행
            proc = subprocess.Popen(
                cmd,
                cwd=script_dir,
                stdout=None, # 실시간 터미널 출력 유지
                stderr=None
            )
            processes.append(proc)
            print(f" >> [Client #{client_id}] 프로세스 시작 완료 (PID: {proc.pid})")
            time.sleep(0.3)  # 서버 연결 타이밍 안정화

        print("\n" + "=" * 70)
        print(f" >> 총 {args.num_clients}대의 시뮬레이션 클라이언트가 정상 실행되었습니다.")
        print(f" >> 서버({args.server})와 학습 라운드를 진행합니다. (종료하려면 Ctrl+C)")
        print("=" * 70 + "\n")

        # 모든 프로세스 종료 대기
        for p in processes:
            p.wait()

        print("\n[Simulator] 모든 시뮬레이션 클라이언트 학습이 완료되었습니다.")

    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
