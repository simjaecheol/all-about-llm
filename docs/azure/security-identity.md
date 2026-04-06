---
layout: default
title: 보안 & 인증 (Key Vault, Managed Identity)
parent: Azure Cloud 배포 (AI 프로덕션)
nav_order: 4
---

# 보안 & 인증: Key Vault, Managed Identity, RBAC

## 개요

프로덕션 AI 서비스에서 가장 흔한 보안 실수는 **API 키를 코드나 환경변수 파일에 직접 기입**하는 것입니다. Azure에서는 다음 세 가지 원칙을 통해 이를 방지합니다.

1. **Secrets는 Key Vault에**: API 키, DB 연결 문자열, 인증서를 코드에서 분리하여 중앙 관리
2. **비밀번호 없는 인증 (Managed Identity)**: Service Principal의 `client_secret` 없이 Azure 서비스 간 신뢰 관계로 인증
3. **최소 권한 원칙 (RBAC)**: 각 서비스/사람이 꼭 필요한 권한만 갖도록 세밀하게 제어

```mermaid
graph TB
    Dev["👨‍💻 개발자\n(Entra ID 계정)"]
    GHCI["GitHub Actions\n(Workload Identity)"]
    AKSPod["AKS Pod\n(Workload Identity)"]

    subgraph "Azure 신뢰 체계"
        EntraID["Microsoft Entra ID\n(중앙 ID 관리)"]
        MI_App["Managed Identity\n(ai-app-identity)"]
        MI_GH["Managed Identity\n(github-actions-identity)"]
    end

    subgraph "보호 대상 리소스"
        KV["Azure Key Vault\n(Secrets / Certs)"]
        ACR["Azure Container Registry"]
        AKS["AKS Cluster"]
    end

    Dev -->|RBAC 역할로| KV
    Dev -->|RBAC 역할로| AKS
    GHCI -->|OIDC 토큰| EntraID
    AKSPod -->|OIDC 토큰| EntraID
    EntraID -->|액세스 토큰 발급| MI_App
    EntraID -->|액세스 토큰 발급| MI_GH
    MI_App -->|"Key Vault Secrets User"| KV
    MI_GH -->|"AcrPush"| ACR
```

---

## 1. Azure Key Vault

**Azure Key Vault**는 다음 세 가지 유형의 민감 정보를 안전하게 저장하고 버전 관리합니다.

| 유형 | 용도 | 예시 |
| :--- | :--- | :--- |
| **Secrets** | API 키, 비밀번호, 연결 문자열 | `OPENAI_API_KEY`, `DB_PASSWORD` |
| **Keys** | 암/복호화용 암호화 키 | 데이터 암호화, 서명 생성 |
| **Certificates** | TLS/SSL 인증서 | App Gateway에 연결되는 HTTPS 인증서 |

### Key Vault 생성 및 Secret 등록

```bash
# Key Vault 생성
az keyvault create \
  --name my-ai-kv \
  --resource-group my-rg \
  --location koreacentral \
  --enable-rbac-authorization true   # RBAC 모드 사용 (Access Policy 방식 지양)

# Secret 등록
az keyvault secret set \
  --vault-name my-ai-kv \
  --name "openai-api-key" \
  --value "sk-..."

az keyvault secret set \
  --vault-name my-ai-kv \
  --name "db-connection-string" \
  --value "postgresql://user:pass@host:5432/db"
```

### AKS에서 Key Vault Secret 사용: CSI Driver

AKS의 **Secrets Store CSI Driver**는 Key Vault의 Secret을 파드 내 파일 또는 환경변수로 마운트합니다. 파드가 재시작되면 최신 Secret 값으로 자동 갱신됩니다.

```yaml
# secretproviderclass.yaml
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: ai-app-secrets
  namespace: production
spec:
  provider: azure
  parameters:
    usePodIdentity: "false"
    clientID: <managed-identity-client-id>   # Workload Identity와 연결
    keyvaultName: my-ai-kv
    tenantId: <tenant-id>
    objects: |
      array:
        - |
          objectName: openai-api-key
          objectType: secret
        - |
          objectName: db-connection-string
          objectType: secret
  secretObjects:
    - secretName: ai-app-env-secrets   # K8s Secret으로도 동기화
      type: Opaque
      data:
        - objectName: openai-api-key
          key: OPENAI_API_KEY
        - objectName: db-connection-string
          key: DATABASE_URL
```

```yaml
# deployment.yaml (일부) - Secret을 환경변수로 주입
spec:
  containers:
    - name: ai-app
      image: myacrregistry.azurecr.io/ai-app:latest
      envFrom:
        - secretRef:
            name: ai-app-env-secrets   # CSI Driver가 생성한 K8s Secret
      volumeMounts:
        - name: secrets-store
          mountPath: "/mnt/secrets"
          readOnly: true
  volumes:
    - name: secrets-store
      csi:
        driver: secrets-store.csi.k8s.io
        readOnly: true
        volumeAttributes:
          secretProviderClass: ai-app-secrets
```

---

## 2. Managed Identity & Workload Identity

**Managed Identity**는 Azure가 관리하는 ID로, 서비스 주체(Service Principal)의 `client_id` + `client_secret`을 직접 관리할 필요 없이 Azure 리소스 간 신뢰 기반으로 인증합니다.

### Managed Identity 종류

| 종류 | 설명 | 사용 사례 |
| :--- | :--- | :--- |
| **System-assigned** | Azure 리소스(VM, AKS)에 자동 생성·소멸 | 단순 1:1 연결 |
| **User-assigned** | 사용자가 독립적으로 생성, 여러 리소스에 공유 | **AI 앱 권장** (재사용 가능, 생명주기 독립) |

### Workload Identity 전체 설정 흐름

```bash
# 1. User-assigned Managed Identity 생성
az identity create \
  --name ai-app-identity \
  --resource-group my-rg

# 클라이언트 ID 확인 (Pod에 주입할 값)
CLIENT_ID=$(az identity show --name ai-app-identity --resource-group my-rg --query clientId -o tsv)

# 2. Key Vault에 역할 부여
az role assignment create \
  --assignee $CLIENT_ID \
  --role "Key Vault Secrets User" \
  --scope /subscriptions/.../vaults/my-ai-kv

# ACR에 이미지 Pull 권한 부여
az role assignment create \
  --assignee $CLIENT_ID \
  --role "AcrPull" \
  --scope /subscriptions/.../registries/myacrregistry

# 3. AKS OIDC Issuer URL 확인
OIDC_URL=$(az aks show --name my-aks --resource-group my-rg --query oidcIssuerProfile.issuerUrl -o tsv)

# 4. Federated Credential 등록 (K8s ServiceAccount와 연결)
az identity federated-credential create \
  --name ai-app-fed \
  --identity-name ai-app-identity \
  --resource-group my-rg \
  --issuer $OIDC_URL \
  --subject "system:serviceaccount:production:ai-app-sa"

# 5. K8s ServiceAccount 생성 (Managed Identity 클라이언트 ID 주석 추가)
kubectl create serviceaccount ai-app-sa --namespace production
kubectl annotate serviceaccount ai-app-sa \
  --namespace production \
  azure.workload.identity/client-id=$CLIENT_ID
```

```yaml
# deployment.yaml에서 ServiceAccount 지정
spec:
  serviceAccountName: ai-app-sa
  template:
    metadata:
      labels:
        azure.workload.identity/use: "true"   # Workload Identity 활성화
```

---

## 3. Azure RBAC: 역할 기반 접근 제어

**Azure RBAC**은 리소스에 대한 접근 권한을 **누가(Who)**, **무엇을(What)**, **어디서(Where)** 할 수 있는지 정의합니다. 최소 권한 원칙(Principle of Least Privilege)을 적용해야 합니다.

### AI 서비스 배포에서 주요 역할 정의

| 역할 | 부여 대상 | 범위 | 설명 |
| :--- | :--- | :--- | :--- |
| `AcrPush` | GitHub Actions MI | ACR | 이미지 빌드 후 푸시 |
| `AcrPull` | App MI, AKS | ACR | 이미지 배포 시 Pull |
| `Key Vault Secrets User` | App MI | Key Vault | Secret 읽기 (쓰기 불가) |
| `Key Vault Secrets Officer` | 담당 개발자 | Key Vault | Secret 생성·수정·삭제 |
| `Contributor` | DevOps 엔지니어 | Resource Group | 인프라 변경 |
| `Reader` | 일반 개발자 | Resource Group | 상태 조회만 |

### 역할 할당 명령어

```bash
# 개발자에게 Key Vault Secret 관리 권한 부여
az role assignment create \
  --assignee user@company.com \
  --role "Key Vault Secrets Officer" \
  --scope /subscriptions/.../vaults/my-ai-kv

# 특정 그룹에게 AKS 읽기 권한 부여
az role assignment create \
  --assignee <group-object-id> \
  --role "Azure Kubernetes Service Cluster User Role" \
  --scope /subscriptions/.../managedClusters/my-aks
```

---

## 4. Microsoft Entra ID (구 Azure AD)

**Microsoft Entra ID**는 Azure의 중앙 ID 및 인증 서비스입니다. 개발자와 서비스의 모든 인증 흐름이 Entra ID를 통해 이루어집니다.

### AKS와 Entra ID 통합 (RBAC)

```bash
# AKS에 Entra ID 통합 기반 RBAC 활성화
az aks update \
  --resource-group my-rg \
  --name my-aks \
  --enable-azure-rbac \
  --enable-aad

# 특정 Entra ID 그룹에게 AKS 클러스터 관리자 권한 부여
az role assignment create \
  --assignee <entra-group-id> \
  --role "Azure Kubernetes Service RBAC Cluster Admin" \
  --scope /subscriptions/.../managedClusters/my-aks
```

이후 개발자는 자신의 Entra ID 계정으로 AKS에 로그인합니다.

```bash
az aks get-credentials --resource-group my-rg --name my-aks
# → 브라우저 팝업 또는 디바이스 코드로 Entra ID 로그인
kubectl get pods -n production
```

---

## 관련 문서

- **[AKS 설계 및 운영](./aks.md)**: Workload Identity가 실제 적용되는 AKS 클러스터
- **[ACR + CI/CD](./acr-cicd.md)**: GitHub Actions의 Workload Identity Federation
- **[네트워크](./networking.md)**: Key Vault, ACR을 VNet 안에 격리하는 Private Endpoint
