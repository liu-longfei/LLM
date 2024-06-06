# git rebase

Git rebase 是一种整理提交历史的手段，它可以将一系列更改从一个分支转移到另一个分支的基础上。这通常用于保持干净的线性历史。如果在 rebase 过程中出现冲突，你需要手动解决这些冲突，然后继续 rebase 操作。

以下是 git rebase 的基本流程，包括冲突解决步骤：

### 1. 开始 Rebase

假设你在一个名为 `feature` 的分支上工作，并想要将 `main`（或 `master`）分支上的最新更改整合到 `feature` 分支上。

首先，确保你的 `main` 分支是最新的：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git checkout main
git pull
</code></pre>

然后，从你的 `feature` 分支开始 rebase：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git checkout feature
git rebase main
</code></pre>

### 2. 解决冲突

如果在 rebase 过程中遇到冲突，git 会停止并允许你解决冲突。git 会在输出中标明冲突文件。

#### 查看冲突文件

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git status
</code></pre>

冲突的文件会在输出中显示为 “both modified”。

#### 手动解决冲突

打开冲突文件，查找以下标记的部分：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang"></span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body "><<<<<<< HEAD
// 你的 `feature` 分支上的更改
=======
// `main` 分支上的更改
>>>>>>> main
</code></pre>

编辑文件以解决冲突，删除上述标记，并保留你想要的代码。

#### 标记冲突文件为已解决

一旦解决冲突，你需要将这些文件标记为已解决状态：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git add <冲突文件>
</code></pre>

#### 继续 Rebase

解决所有冲突并标记为已解决后，继续 rebase 过程：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git rebase --continue
</code></pre>

如果你决定不继续 rebase 并想要返回到 rebase 开始前的状态：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git rebase --abort
</code></pre>

### 3. 完成 Rebase

解决所有冲突并完成 rebase 后，如果你的 `feature` 分支是远程跟踪的，你可能需要使用 `git push --force` 或 `git push --force-with-lease` 将更改推送到远程仓库，因为 rebase 会改变历史。

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git push --force
</code></pre>

或者，如果需要更安全地推送（确保不会覆盖远程分支上的其他人的工作），可以使用：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git push --force-with-lease
</code></pre>

### 注意事项

* **不要在共享分支上使用 rebase。** 如果其他人也在该分支上工作，rebase 会重写公共历史，这可能会导致其他人的仓库进入混乱状态。
* **理解你正在重写历史。** 在执行 rebase 操作时，你实际上是在创建一系列新的提交，并替换掉旧的提交。如果不小心，可能会丢失更改。

以上就是 git rebase 的基本流程及解决冲突的步骤。如果你是 git 的新手，建议在更熟悉这个工具之前，先在不会对其他人造成影响的个人分支上练习使用 rebase。

# git merge

当你在Git中处理多个分支时，经常需要将一个分支的改动合并到另一个分支。以下是使用`git merge`的基本流程，包括如何解决合并冲突。

### 1. 确保你在正确的分支上

首先，你需要确保自己在想要合并**进**的分支上。例如，如果你想将分支`feature`的改动合并到`main`分支，你应该首先切换到`main`分支：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git checkout main
</code></pre>

### 2. 合并分支

然后，使用`git merge`命令将`feature`分支合并进`main`分支：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git merge feature
</code></pre>

这将尝试自动合并`feature`分支的更改到`main`分支。如果没有冲突，Git会自动创建一个新的合并提交，除非所有更改都已经是当前分支的快进（fast-forward）。

### 3. 解决合并冲突

如果存在冲突，Git将无法自动完成合并，并会提示你解决这些冲突。你可以使用`git status`查看哪些文件产生了冲突：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git status
</code></pre>

冲突的文件会被标记为“未合并”。

**示例冲突解决：** 假设在文件`example.txt`中有冲突，打开这个文件，你会看到类似下面的内容：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">plaintext</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body plaintext"><<<<<<< HEAD
这是main分支的内容
=======
这是feature分支的内容
>>>>>>> feature
</code></pre>

你需要决定保留`main`分支的内容、`feature`分支的内容，或者两者的组合。编辑文件，解决这些冲突，然后保存文件。

### 4. 添加解决了冲突的文件

解决冲突后，你需要将这些文件标记为已解决冲突：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git add example.txt
</code></pre>

### 5. 完成合并

一旦解决了所有冲突并且添加了所有文件，你可以通过创建一个新的合并提交来完成合并过程：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git commit
</code></pre>

Git会打开一个编辑器，让你输入合并提交的消息。通常，它会自动生成一条合并提交消息，说明了合并了哪些分支。你可以直接使用这条消息，或者根据需要进行修改。

### 6. 推送更改

最后，推送你的更改到远程仓库：

<pre class="code-block-wrapper"><div class="code-block-header"><span class="code-block-header__lang">bash</span><span class="code-block-header__copy">复制代码</span></div><code class="hljs code-block-body bash">git push
</code></pre>

这将同步你的工作到远程服务器。

通过以上步骤，你可以有效地使用`git merge`进行分支合并，并解决可能出现的合并冲突。使用这些命令可以确保你的项目分支同步和整合不同开发者的工作。
