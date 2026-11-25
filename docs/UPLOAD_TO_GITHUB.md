# Upload this code to an existing GitHub repository

Upload the **contents of the `github` folder** to the repository root. The root
should contain `README.md`, `train.py`, `hintnet/`, `configs/` and `annotation/`.
Include `.gitignore`. No model-weight binaries are included.

## From a Linux terminal

Replace `/path/to/HINT-Net` with the parent directory containing this release
folder, and `YOUR_USERNAME/YOUR_REPOSITORY` with your repository address. Clone first
so an existing README or commit history is handled normally.

```bash
cd /path/to/HINT-Net
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git github-upload
cp -a github/. github-upload/
cd github-upload
git add .
git commit -m "Add HINT-Net 336 code and annotation XML"
git push -u origin HEAD
```

For HTTPS authentication, use a Git credential helper, GitHub CLI browser login,
or a personal access token when Git asks for a password. A GitHub account
password cannot authenticate a Git push. Do not put a token into the commands
or repository files.

## From the GitHub website

1. Open the repository and choose **Add file → Upload files**. For an empty
   repository, use the upload link in its quick-setup page.
2. Select the files and subfolders **inside** the release folder, preserving
   their directory structure and including `.gitignore`.
3. Confirm that `README.md` and `annotation/label_studio_config.xml` appear at
   their intended paths, enter a commit message and commit/propose the upload.

The XML is the authors' supplied labeling-interface configuration. It is not a
trained model or per-video annotation export.

References: [upload files](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository),
[clone a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository),
[GitHub authentication](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github).
