$ErrorActionPreference='Stop'
$app=New-Object -ComObject PowerPoint.Application
$src=[IO.Path]::GetFullPath('E:/process/domain/outputs/query_vertical_ppt/Figure3_vertical_queries.pptx')
$dst=[IO.Path]::GetFullPath('E:/process/domain/outputs/query_vertical_ppt/Figure3_vertical_queries.pdf')
$deck=$app.Presentations.Open($src,-1,0,0)
$deck.SaveAs($dst,32)
$deck.Close()
'Exported vector PDF'
