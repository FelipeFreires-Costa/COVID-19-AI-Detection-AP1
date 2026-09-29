# Subir o reposit&oacute;rio para o GitHub (5 minutos)

O reposit&oacute;rio local j&aacute; est&aacute; criado e com o primeiro commit feito.

1. Acesse <https://github.com/new>, use o nome `tp1-g4-covid19-baseline` e marque **Public**. **N&atilde;o** marque "Add a README".
2. No PowerShell, troque `SEU_USUARIO` pelo seu usu&aacute;rio do GitHub e rode:

```powershell
cd "C:\Users\Felipe\Documents\estudos\AP1 topicos especiais\tp1-g4-covid19-baseline"
$u = "SEU_USUARIO"
Get-ChildItem -Recurse -File -Include *.tex,*.ipynb,*.md,*.html | Where-Object { $_.FullName -notmatch "\\.venv\\" } | ForEach-Object { $c = Get-Content $_.FullName -Raw; if ($c -match "SEU_USUARIO") { $c.Replace("SEU_USUARIO", $u) | Set-Content -NoNewline -Encoding ascii $_.FullName } }
git add -A
git commit -m "Link do repositorio"
git remote add origin "https://github.com/$u/tp1-g4-covid19-baseline.git"
git push -u origin main
```

3. Em *Settings* &rarr; *Collaborators*, adicione Eduardo e Pedro.

## Depois da execu&ccedil;&atilde;o no Kaggle

```powershell
cd "C:\Users\Felipe\Documents\estudos\AP1 topicos especiais\tp1-g4-covid19-baseline"
Expand-Archive -Force "$env:USERPROFILE\Downloads\tp1_results.zip" .
git add results splits paper
git commit -m "Resultados da execucao completa no Kaggle"
git push
```
