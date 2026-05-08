# 将 my_learning 提交到 GitHub

## 背景

当前仓库的 `origin` 指向官方 `isaac-sim/IsaacLab`，你无写入权限。
正确做法是将官方仓库 **Fork** 到你自己的 GitHub 账号，再把本地改动推送到你的 Fork。

---

## 第一步：配置 git 用户信息

```bash
git config --global user.name  "你的名字"
git config --global user.email "hit.lgcx@gmail.com"
```

---

## 第二步：在 GitHub 创建 Fork

浏览器打开 https://github.com/isaac-sim/IsaacLab，点右上角 **Fork** 按钮。

Fork 完成后你会得到：`https://github.com/<你的用户名>/IsaacLab`

---

## 第三步：把 Fork 添加为本地 remote

```bash
cd /home/hit-lgc/HomeWorkspace/unitree_rl_lab/IsaacLab

# 添加你的 fork 为 myfork remote（origin 保留指向上游，用于同步更新）
git remote add myfork https://github.com/<你的用户名>/IsaacLab.git

# 确认 remote 配置
git remote -v
# 预期输出：
#   myfork  https://github.com/<你的用户名>/IsaacLab.git (fetch)
#   myfork  https://github.com/<你的用户名>/IsaacLab.git (push)
#   origin  https://github.com/isaac-sim/IsaacLab.git    (fetch)
#   origin  https://github.com/isaac-sim/IsaacLab.git    (push)
```

---

## 第四步：创建 .gitignore（排除训练日志和模型权重）

在 `scripts/my_learning/` 目录下创建 `.gitignore`：

```bash
cat > /home/hit-lgc/HomeWorkspace/unitree_rl_lab/IsaacLab/scripts/my_learning/.gitignore << 'EOF'
# 训练日志和模型权重（体积大，不提交）
logs/

# Python 缓存
__pycache__/
*.pyc
*.pyo

# IDE 配置
.vscode/
.idea/
EOF
```

---

## 第五步：暂存并提交你的改动

```bash
cd /home/hit-lgc/HomeWorkspace/unitree_rl_lab/IsaacLab

# 查看待提交文件
git status

# 只添加你自己的文件（不要 git add . 避免误提交）
git add scripts/my_learning/
git add CLAUDE.md

# 提交
git commit -m "add my_learning: Go2 locomotion training scripts (steps 1-19)"
```

---

## 第六步：创建并推送到你的分支

```bash
# 基于 main 创建个人分支
git checkout -b my-learning

# 推送到你的 fork（首次推送用 -u 建立跟踪关系）
git push -u myfork my-learning
```

推送成功后访问：`https://github.com/<你的用户名>/IsaacLab/tree/my-learning`

---

## 日常工作流（后续新增脚本时）

```bash
# 修改或新增脚本后
git add scripts/my_learning/step_XX_xxx.py
git commit -m "add step XX: 描述"
git push myfork my-learning
```

---

## 同步上游官方更新

```bash
# 拉取官方最新代码
git fetch origin

# 将上游更新合并到你的分支
git rebase origin/main my-learning

# 推送到你的 fork（rebase 后需要 --force-with-lease）
git push --force-with-lease myfork my-learning
```

---

## 使用 HTTPS 推送时需要 Token 认证

GitHub 已禁用密码登录，需要使用 Personal Access Token (PAT)：

1. 浏览器打开 https://github.com/settings/tokens
2. 点击 **Generate new token (classic)**
3. 勾选 `repo` 权限，生成 token（形如 `ghp_xxxxxxxxxxxx`）
4. 推送时用 token 替代密码：

```bash
# 推送时提示输入密码，填入 token 即可
git push -u myfork my-learning
# Username: <你的用户名>
# Password: ghp_xxxxxxxxxxxx   ← 填 token，不是 GitHub 密码
```

或者配置凭据缓存避免每次输入：

```bash
git config --global credential.helper store
# 首次推送输入一次后自动保存到 ~/.git-credentials
```

---

## 推荐安装 gh CLI（可选，简化操作）

```bash
# Ubuntu/Debian
sudo apt install gh

# 登录
gh auth login

# 之后可以直接用命令 Fork 和创建 PR
gh repo fork isaac-sim/IsaacLab --clone=false
```
