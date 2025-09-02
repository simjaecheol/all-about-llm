#!/usr/bin/env python3
"""
Black formatter 환경 설정 자동화 스크립트
"""

import os
import subprocess


def run_command(command, description):
    """명령어를 실행하고 결과를 출력합니다."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} 완료")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 실패: {e}")
        if e.stderr:
            print(f"에러: {e.stderr}")
        return False


def main():
    print("🚀 Black formatter 환경 설정을 시작합니다...")

    # pip 업그레이드
    run_command("python -m pip install --upgrade pip", "pip 업그레이드")

    # 필요한 패키지 설치
    run_command("pip install -r requirements.txt", "필요한 패키지 설치")

    # pre-commit 설치
    run_command("pre-commit install", "pre-commit hooks 설치")

    # 현재 Python 파일들에 Black 적용
    print("\n🔄 기존 Python 파일들에 Black formatter 적용...")
    python_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    if python_files:
        print(f"📁 {len(python_files)}개의 Python 파일을 찾았습니다.")
        for file_path in python_files:
            print(f"  - {file_path}")

        # Black 적용
        run_command("black .", "모든 Python 파일에 Black formatter 적용")
    else:
        print("📁 Python 파일을 찾을 수 없습니다.")

    print("\n🎉 Black formatter 환경 설정이 완료되었습니다!")
    print("\n📋 사용 방법:")
    print("1. VS Code에서 Python 파일을 열면 자동으로 포맷팅됩니다")
    print("2. 파일 저장 시 자동으로 Black이 적용됩니다")
    print("3. Git commit 시 pre-commit hooks가 자동으로 실행됩니다")
    print("4. 수동으로 실행하려면: black <파일명>")


if __name__ == "__main__":
    main()
