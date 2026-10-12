$ErrorActionPreference='Stop'
function C($h){[Convert]::ToInt32($h.Substring(0,2),16)+256*[Convert]::ToInt32($h.Substring(2,2),16)+65536*[Convert]::ToInt32($h.Substring(4,2),16)}
$app=New-Object -ComObject PowerPoint.Application
$deck=$app.Presentations.Add(0)
$deck.PageSetup.SlideWidth=300
$deck.PageSetup.SlideHeight=212
$s=$deck.Slides.Add(1,12)
$s.FollowMasterBackground=0
$s.Background.Fill.ForeColor.RGB=(C 'FFFFFF')
function Text($value,$x,$y,$w,$h,$size=9,$color='263640',$bold=$false,$math=$false){
 $a=$s.Shapes.AddTextbox(1,$x,$y,$w,$h)
 $a.TextFrame.MarginLeft=0;$a.TextFrame.MarginRight=0;$a.TextFrame.MarginTop=0;$a.TextFrame.MarginBottom=0
 $a.TextFrame.AutoSize=0;$a.TextFrame.WordWrap=0;$a.TextFrame.VerticalAnchor=3
 $t=$a.TextFrame.TextRange;$t.Text=$value;$t.Font.Name='Arial';if($math){$t.Font.Name='Cambria Math'}
 $t.Font.Size=[single]$size;$t.Font.Color.RGB=(C $color);$t.Font.Bold=[int]$bold*-1;$t.ParagraphFormat.Alignment=2
 return $a
}
function Box($value,$x,$y,$w,$h,$fill='F2F3F4',$size=9,$math=$false,$stroke='A5ADB2'){
 $a=$s.Shapes.AddShape(5,$x,$y,$w,$h);$a.Adjustments.Item(1)=.08
 $a.Fill.ForeColor.RGB=(C $fill);$a.Line.ForeColor.RGB=(C $stroke);$a.Line.Weight=.55
 $a.TextFrame.MarginLeft=1;$a.TextFrame.MarginRight=1;$a.TextFrame.MarginTop=0;$a.TextFrame.MarginBottom=0
 $a.TextFrame.AutoSize=0;$a.TextFrame.WordWrap=0;$a.TextFrame.VerticalAnchor=3
 $t=$a.TextFrame.TextRange;$t.Text=$value;$t.Font.Name='Arial';if($math){$t.Font.Name='Cambria Math'}
 $t.Font.Size=[single]$size;$t.Font.Color.RGB=(C '263640');$t.ParagraphFormat.Alignment=2
 return $a
}
function Line($x,$y,$xx,$yy,$arrow=$false,$color='53636D'){
 $a=$s.Shapes.AddLine($x,$y,$xx,$yy);$a.Line.ForeColor.RGB=(C $color);$a.Line.Weight=.65
 if($arrow){$a.Line.EndArrowheadStyle=3;$a.Line.EndArrowheadLength=1;$a.Line.EndArrowheadWidth=1}
 return $a
}

$orange='FBE8CA';$green='DCEEDC';$blue='EDF2F7'
# Legend only; each predictor below has its own input and path.
$null=Box '' 12 7 17 11 $orange 9
$null=Text 'Segmentation query' 33 5 109 15 8.6
$null=Box '' 156 7 17 11 $green 9 $false '79A582'
$null=Text 'Function query' 177 5 105 15 8.6
$null=Line 7 27 293 27 $false 'BCC4C9'
foreach($x in @(100,200)){
 $a=Line $x 35 $x 177 $false 'B8C1C7';$a.Line.DashStyle=4
}

# (a) Shared query: one vector inherited by both regions.
$null=Box 'qᵢ' 33 38 34 23 $orange 13 $true
$null=Line 50 63 50 101 $true
$null=Box "Function`nhead" 13 104 74 37 'F2F3F4' 9.3
$null=Line 50 143 50 153
$null=Line 29 153 71 153
$null=Line 29 153 29 163 $true
$null=Line 71 153 71 163 $true
$null=Text 'p₁ = p₂' 12 165 76 20 12 '263640' $false $true
$null=Text '(a) Shared query' 3 185 94 14 8.8

# (b) Region MLP: independent feature vectors with shared classifier weights.
foreach($r in @(@(113,130,'h₁','p₁'),@(155,172,'h₂','p₂'))){
 $x=[single]$r[0];$cx=[single]$r[1]
 $null=Box $r[2] $x 38 34 23 $blue 12 $true
 $null=Line $cx 63 $cx 101 $true
 $null=Line $cx 143 $cx 163 $true
 $null=Text $r[3] ($cx-17) 165 34 20 12 '263640' $false $true
}
$null=Box "Shared`nMLP" 113 104 76 37 'F2F3F4' 9.3
$null=Text '(b) Region MLP' 102 185 96 14 8.8

# (c) QuaSF: one function-query state per region.
foreach($r in @(@(213,230,'h₁','qᶠ₁','p₁'),@(255,272,'h₂','qᶠ₂','p₂'))){
 $x=[single]$r[0];$cx=[single]$r[1]
 $null=Box $r[2] $x 38 34 23 $blue 12 $true
 $null=Line $cx 63 $cx 70 $true
 $null=Box $r[3] $x 70 34 23 $green 12 $true '79A582'
 $null=Line $cx 95 $cx 101 $true
 $null=Line $cx 143 $cx 163 $true
 $null=Text $r[4] ($cx-17) 165 34 20 12 '263640' $false $true
}
$null=Box "Function`ndecoder" 213 104 76 37 'F2F3F4' 9.3
$null=Text '(c) QuaSF' 203 185 94 14 8.8
$s.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text='Figure 3 concept, based on QuaSF manuscript section 3.4, Overleaf project 6a0db889e605477ab8e23124. Three independent vertical panels following the user-provided reference layout. Shared query predicts one vector from source segmentation query qi and assigns it to both regions. Region MLP directly classifies pooled regional features h1,h2. QuaSF forms individual function queries qf1,qf2 before function decoding. Both regional predictors use the same masks and mean/max ESM2 features. Shared model weights are not separate models. Pooling, region generation and query-construction details are omitted to emphasize the functional paths. Arrows indicate computation. All slide elements are native editable objects.'
$out='E:/process/domain/outputs/query_vertical_ppt'
$pptPath=[IO.Path]::GetFullPath((Join-Path $out 'Figure3_vertical_queries.pptx'))
$pngPath=[IO.Path]::GetFullPath((Join-Path $out 'Figure3_vertical_queries.png'))
$deck.SaveAs($pptPath,24)
$s.Export($pngPath,'PNG',1500,1060)
$deck.Close()
'Created vertical editable figure'
