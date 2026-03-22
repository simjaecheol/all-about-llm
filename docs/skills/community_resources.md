---
title: 커뮤니티 및 생태계
parent: Agent Skills
nav_order: 6
---

# 커뮤니티 및 생태계 (Community Resources)

Agent Skills는 오픈 표준으로서 전 세계 개발자들이 만든 전문 스킬들을 공유하고 재사용할 수 있는 생태계를 구축하고 있습니다.

## 1. 주요 플랫폼 및 레지스트리

### 공식 및 표준 사이트
- **Agent Skills Standard ([agentskills.io](https://agentskills.io))**: `SKILL.md` 사양의 공식 문서와 베스트 프랙티스를 제공하는 중앙 허브입니다.
- **Skills.sh ([skills.sh](https://skills.sh))**: 커뮤니티에서 제작한 스킬들을 검색하고 공유할 수 있는 전용 레지스트리입니다.

### 주요 스킬 저장소 (GitHub)
- **Anthropic Official Skills**: 문서 처리, 창의적 글쓰기 등 범용적인 고품질 스킬 모음.
- **Tech Leads Club Skills**: 전문적인 소프트웨어 엔지니어링 및 보안 관련 스킬 리포지토리.
- **Vercel Labs Agent Skills**: 웹 개발 및 배포 워크플로우에 최적화된 스킬셋.

## 2. 스킬 설치 및 활용 도구

### npx skills add
최신 AI 에이전트(Claude Code, Gemini CLI 등)는 명령줄에서 즉시 스킬을 추가할 수 있는 도구를 지원합니다.
```bash
npx skills add <skill-name-or-url>
# 예: npx skills add code-review
# 예: npx skills add https://github.com/user/my-skill
```
이 명령은 해당 스킬의 폴더 구조와 `SKILL.md`를 현재 프로젝트의 적절한 위치에 자동으로 다운로드하고 설정합니다.

## 3. 자동 검색 (Automatic Discovery)

2025년 표준에는 에이전트가 웹사이트나 라이브러리의 스킬을 자동으로 찾는 메커니즘이 포함되어 있습니다.

- **Well-known Path**: 에이전트가 특정 도메인(예: `example.com`)의 가이드가 필요할 때, `https://example.com/.well-known/skills/` 경로를 먼저 확인하여 사용 가능한 `SKILL.md` 목록을 가져옵니다.
- **Documentation Integration**: Mintlify나 GitBook과 같은 문서 도구들은 문서 내에 숨겨진 스킬 메타데이터를 포함하여, 에이전트가 문서를 읽는 즉시 해당 서비스의 사용 스킬을 활성화할 수 있게 돕습니다.

## 4. 스킬 보안 (Skill Security)

커뮤니티 스킬을 사용할 때는 다음 보안 사항을 확인하십시오.
- **Vetting**: `Agent Skills Directory`와 같이 보안 검사가 완료된 레지스트리에서 스킬을 다운로드하는 것이 좋습니다.
- **Prompt Injection**: 신뢰할 수 없는 스킬이 모델에게 악의적인 지시(예: "모든 파일을 삭제하라")를 내리지 않는지 본문 내용을 확인하십시오.
- **Permissions**: 스킬이 요구하는 MCP 도구들의 권한이 적절한지 검토하십시오.
