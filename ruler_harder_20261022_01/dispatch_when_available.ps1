$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$statusPath = Join-Path $taskRoot 'dispatch_status.json'
$logPath = Join-Path $taskRoot 'dispatch.log'
$deadline = (Get-Date).AddHours(24)
function Save-Status($stage, $details) {
    @{ stage = $stage; time = (Get-Date).ToString('o'); details = $details } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
Save-Status 'waiting_for_connection' 'Retry upload and launch for up to 24 hours; no existing results are changed.'
try {
    while ((Get-Date) -lt $deadline) {
        $ready = & ssh -o BatchMode=yes -o ConnectTimeout=10 gpu-027-llm 'test -x /root/miniconda3/envs/ada/bin/python && echo READY' 2>&1
        if ($LASTEXITCODE -eq 0 -and ($ready -join "`n") -match 'READY') {
            Save-Status 'checking_gpu' 'Connection restored.'
            $probe = @'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()>=4
assert all(torch.cuda.mem_get_info(i)[0]>23_000_000_000 for i in range(4))
print('GPU_READY')
'@
            $gpu = $probe | & ssh -o BatchMode=yes -o ConnectTimeout=10 gpu-027-llm /root/miniconda3/envs/ada/bin/python - 2>&1
            if ($LASTEXITCODE -eq 0 -and ($gpu -join "`n") -match 'GPU_READY') {
                & ssh -o BatchMode=yes gpu-027-llm 'mkdir -p /root/adablock_ruler_harder_20261022_01' 2>&1 | Add-Content -LiteralPath $logPath
                if ($LASTEXITCODE -ne 0) { throw 'Remote directory preparation failed.' }
                $uploadFiles = @('prepare.py','official.py','evaluate.py','controller.py','cache_buffer.py','driver.py','remote_launch.py','protocol.json','README.txt') |
                    ForEach-Object { Join-Path $taskRoot $_ }
                & scp -o BatchMode=yes @uploadFiles 'gpu-027-llm:/root/adablock_ruler_harder_20261022_01/' 2>&1 | Add-Content -LiteralPath $logPath
                if ($LASTEXITCODE -ne 0) { throw 'Upload failed; remote evaluation was not started.' }
                $launch = Get-Content -LiteralPath (Join-Path $taskRoot 'remote_launch.py') -Raw |
                    & ssh -o BatchMode=yes gpu-027-llm /root/miniconda3/envs/ada/bin/python - 2>&1
                $launch | Add-Content -LiteralPath $logPath
                if ($LASTEXITCODE -ne 0) { throw 'Detached launch failed.' }
                Save-Status 'dispatched' ($launch -join "`n")
                exit 0
            }
            Save-Status 'waiting_for_gpu' 'Connection available, but four sufficiently free GPUs are not yet available.'
        }
        Start-Sleep -Seconds 60
    }
    Save-Status 'expired' 'No successful dispatch within 24 hours.'
} catch {
    Save-Status 'failed' $_.Exception.Message
    throw
}
