param(
  [ValidateSet("local","docker")] [string]$Mode = "local",
  [ValidateSet("site_sitemap","site_crawl","focused_search")] [string]$Spider = "focused_search",
  [Parameter(ValueFromRemainingArguments=$true)] [string[]]$Rest
)

# Example usage:
#   .\start.ps1 local focused_search -a q="bedrock claude 3.7" -a seeds="https://aws.amazon.com,https://docs.aws.amazon.com" -a max_pages=120 -a top_k=25
#   .\start.ps1 docker focused_search -a q="kubernetes hpa keda tutorial" -a seeds="https://keda.sh,https://kubernetes.io" -a max_pages=80

$extra = $Rest -join ' '

function Ensure-Local {
  if (-not $env:VIRTUAL_ENV -and (Test-Path .\.venv\Scripts\Activate.ps1)) {
    . .\.venv\Scripts\Activate.ps1
  }
  if (Test-Path .\pyproject.toml) {
    python -m pip install -e .
  } elseif (Test-Path .\requirements.txt) {
    pip install -r requirements.txt
  } else {
    Write-Error "No pyproject.toml or requirements.txt found."
    exit 1
  }
  python -m playwright install chromium
}

if ($Mode -eq "local") {
  Ensure-Local
  Write-Host "[local] running spider: $Spider"
  & scrapy crawl $Spider @Rest
}
elseif ($Mode -eq "docker") {
  $image = $env:IMAGE
  if (-not $image) { $image = "webscraper:dev" }
  Write-Host "[docker] building $image ..."
  docker build -t $image .
  Write-Host "[docker] running spider: $Spider"
  docker run --rm -it $image scrapy crawl $Spider @Rest
}
