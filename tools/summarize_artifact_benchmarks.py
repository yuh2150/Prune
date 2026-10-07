"""Build a source-linked report from saved artifacts, without running models."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts' / 'artifacts'
OUT = ROOT / 'artifacts' / 'pruning_benchmark_summary'
OUT.mkdir(exist_ok=True)

def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

def fmt(value, digits=3):
    return 'NR' if value is None else f'{value:.{digits}f}'

def pct(value):
    return 'NR' if value is None else f'{value * 100:.2f}'

def reduction(before, after):
    return None if before in (None, 0) or after is None else 100 * (1 - after / before)

def method(c, r):
    p = c.get('pruning') or {}
    pr = r.get('pruning') or {}
    kind = p.get('pruner', pr.get('pruner_name'))
    crit = p.get('criterion', pr.get('criterion_name'))
    gran = p.get('granularity', pr.get('granularity_name'))
    if kind == 'block_sparse': return 'Block Pruning (weight blocks)'
    if kind == 'depth': return 'Layer / residual-block Pruning'
    if kind == 'nm': return 'N:M (bổ sung)'
    if crit == 'taylor': return 'Taylor-based Gradient Pruning'
    if kind == 'structured': return 'Channel Pruning' if gran == 'channel' else 'Filter Pruning (L1 norm)'
    return {'magnitude':'Magnitude Pruning','snip':'SNIP','grasp':'GraSP','synflow':'SynFlow','lamp':'LAMP'}.get(crit, str(crit))

records = []
quality = []
runtime = []
links = []
for index, path in enumerate(sorted(SOURCE.glob('*/result.json')), 1):
    r = read(path)
    cfgpath = path.parent / 'config.resolved.json'
    if not cfgpath.exists(): cfgpath = path.parent / 'config.json'
    c = read(cfgpath)
    p = c.get('pruning') or {}
    before = r.get('complexity_before') or {}
    after = r.get('complexity_after') or {}
    base = r.get('baseline_metrics') or {}
    final = r.get('final_metrics') or {}
    bench = (r.get('pruning') or {}).get('benchmark') or {}
    ident = f'R{index:02d}'
    name = path.parent.name
    model = (c.get('model') or {}).get('name', (r.get('pruning') or {}).get('model_name', 'NR'))
    classification = 'accuracy' in final or 'accuracy' in base
    group = 'A' if model == 'lenet5_emnist_onnx' else 'B' if model == 'resnet18' else 'C' if model in ('yolov5', 'yolov5s') else 'D'
    flags = []
    if not final: flags.append('thiếu final metrics')
    if any(s in name for s in ('smoke', 'diagnostic', 'normalized', 'capped')): flags.append('diagnostic/smoke')
    if model == 'lenet5': flags.append('baseline khác/không được xác nhận pretrained')
    if p.get('pruner') in ('unstructured', 'block_sparse', 'nm'): flags.append('mask; chưa giảm dense parameters')
    flop_warn = p.get('pruner') in ('unstructured','block_sparse') and after.get('flops') is not None and before.get('flops') and after['flops'] < before['flops'] * .1
    if flop_warn: flags.append('FLOPs đáng ngờ*')
    if bench.get('inference_latency_ms', 0) > 10000: flags.append('latency bất thường; không xếp hạng')
    if group == 'C': flags.append('benchmark NMS=0; validation có NMS')
    ratio = p.get('amount')
    sparsity = after.get('sparsity_pct')
    mode = 'structured' if p.get('pruner') in ('structured','depth') else 'block-sparse mask' if p.get('pruner') == 'block_sparse' else 'N:M mask' if p.get('pruner') == 'nm' else 'unstructured' if p.get('pruner') == 'unstructured' else 'NR'
    desc = f'{mode}; ratio={pct(ratio)}%; sparsity thực={fmt(sparsity,2)}%'
    ft = (c.get('recovery') or {}).get('enabled')
    fttext = 'Không' if ft is False else 'Có (cấu hình; cần kiểm tra stage)' if ft else 'NR'
    status = f'{fttext}; baseline → sau prune/trước FT; sau FT: NR'
    label = f'{ident} · {group} · {model} / {method(c,r)}'
    metrics = ['N/A'] * 4 if classification else [pct(final.get(k)) for k in ('precision','recall','map50','map50_95')]
    quality.append([label, desc, *metrics, status, f'{pct(base.get("accuracy"))} → {pct(final.get("accuracy"))}' if classification else 'N/A', '; '.join(flags) or '—'])
    gf0 = before.get('flops'); gf1 = after.get('flops')
    runtime.append([label, desc, fttext, f'{before.get("params", "NR"):,} → {after.get("params", "NR"):,}' if isinstance(before.get('params'),int) and isinstance(after.get('params'),int) else 'NR', fmt(reduction(before.get('params'),after.get('params')),2), f'{fmt(gf0/1e9 if gf0 is not None else None,6)} → {fmt(gf1/1e9 if gf1 is not None else None,6)}' + ('*' if flop_warn else ''), fmt(reduction(gf0,gf1),2) + ('*' if flop_warn else ''), fmt(bench.get('inference_latency_ms')), fmt(bench.get('total_latency_ms')), fmt(bench.get('fps'),2), '; '.join(flags) or '—'])
    record = {'id':ident,'group':group,'run':name,'config':c,'baseline_metrics':base,'final_metrics':final,'complexity_before':before,'complexity_after':after,'benchmark':bench,'flags':flags,'source':str(path.relative_to(ROOT))}
    records.append(record)
    rel = '../artifacts/' + name
    links.append(f'- **{ident}** [{name}]({rel}/result.json)' + (f' · [config]({rel}/{cfgpath.name})' if cfgpath.exists() else '') + (f' · validation latency: inference={fmt(final.get("inference_ms"))}, NMS={fmt(final.get("nms_ms"))}, total={fmt(final.get("total_ms"))} ms/img' if group == 'C' else ''))

for missing in ['L1 Regularization','L0 Regularization','Attention Head Pruning']:
    quality.append([f'Chưa có kết quả / {missing}','NR','NR','NR','NR','NR','NR','NR','Không có result.json hoàn chỉnh tương ứng'])
    runtime.append([f'Chưa có kết quả / {missing}','NR','NR','NR','NR','NR','NR','NR','NR','NR','Không suy diễn từ config hoặc paper'])

intro = '''# Benchmark pruning từ artifacts

Ngày tổng hợp: 06/10/2026. Nguồn chính: các `result.json` và cấu hình trong `artifacts/artifacts`. Đây là tổng hợp kết quả đã lưu, không phải benchmark mới chạy. Tất cả run có `result.json` đều được giữ, kể cả diagnostic hoặc thiếu dữ liệu; không chọn riêng lần chạy nhanh nhất. Các phương pháp thiếu kết quả được đánh dấu NR.

**Nhóm so sánh:** A = LeNet5 chuyển từ ONNX, custom_47labels, input 1×1×32×32, CPU, baseline accuracy 84%; B = ResNet18, imagenet1k_subset, input 1×3×224×224, CUDA, baseline accuracy 70%; C = YOLOv5s, evaluation dùng `data/coco_500.yaml`, input 1×3×640×640 (kiểm tra từng config); D = model/baseline khác hoặc smoke. Không xếp hạng giữa các nhóm. CPU/GPU model, phiên bản runtime và điều kiện tải không được ghi đủ: **cùng hardware chưa được xác nhận**, kể cả trong nhóm. Config `dataset.name` của YOLO có thể là `mnist` mặc định; dataset detection lấy từ callback evaluation, không lấy tên mặc định này.

**Quy ước:** Quality tính theo %, accuracy được thêm riêng vì bài toán phân loại không có mAP detection. N/A = không áp dụng; NR = không được báo cáo. Ratio là mục tiêu trong config, không đồng nghĩa % parameters bị loại. Sparsity là tỷ lệ zero được báo cáo trên toàn model. Structured có thể báo sparsity 0% vì tensor đã được thu nhỏ. `L1 norm` dùng chấm điểm filter không phải `L1 Regularization`. Block-sparse weight mask và bỏ residual block là hai loại khác nhau, được ghi rõ từng dòng.

**Before/After Fine-tuning:** các run có cấu hình recovery tắt chỉ có baseline và sau prune/trước FT. Không dùng baseline làm số liệu “before FT” rồi gọi final là “after FT”. Không có kết quả sau FT được xác nhận trong tập này. SNIP/GraSP/SynFlow ở đây được áp dụng lên checkpoint đã có; không phải protocol prune-at-initialization rồi train từ đầu như paper gốc.

**Độ tin cậy runtime:** bảng dùng `pruning.benchmark`, giữ nguyên FPS lưu sẵn. `total_latency_ms` chỉ bao gồm inference và post-process được truyền vào benchmark; NMS=0 ở các run YOLO nên không phải latency end-to-end detection, không gồm đọc ảnh/preprocess/I/O. Validation có số đo riêng, ghi tại nguồn từng run, không trộn với benchmark. Không suy FPS từ validation latency. Không có baseline latency tương ứng đủ để kết luận speedup.

**Cảnh báo FLOPs (*):** một số run mask LeNet5 báo 839256 → 7896 FLOPs dù tensor parameters giữ nguyên. Đây là dấu hiệu profiler bỏ sót module sau parametrization, chưa được xác minh bằng profiler độc lập. Bảng giữ số gốc và phép tính reduction có dấu *, **không coi là giảm compute thực tế**. GFLOPs = FLOPs/10^9; framework định nghĩa FLOPs=2×MACs. Parameter/FLOPs reduction chỉ tính từ cặp before/after của chính run khi cả hai có dữ liệu.

'''
qheaders = ['Model / Pruning Method','Loại · ratio / sparsity','Precision (%)','Recall (%)','mAP@0.5 (%)','mAP@0.5:0.95 (%)','Before/After Fine-tuning','Accuracy baseline → prune (%)','Ghi chú']
rheaders = ['Model / Pruning Method','Loại · ratio / sparsity','Fine-tuning','Parameters before → after','Parameter Reduction (%)','GFLOPs before → after','GFLOPs Reduction (%)','Inference Speed (ms/img)','Total Latency (ms/img)','FPS','Ghi chú']

def table(headers, rows):
    def line(row): return '| ' + ' | '.join(str(x).replace('|','/') for x in row) + ' |'
    return '\n'.join([line(headers),line(['---']*len(headers)),*[line(row) for row in rows]])

conclusion = '''
## Đánh giá trade-off

Trong nhóm A, **Filter Pruning theo L1 norm ở ratio 30% là lựa chọn có bằng chứng tốt nhất về accuracy–kích thước cấu trúc**: accuracy 84% → 83.5%, parameters 64851 → 38061 (−41.31%), FLOPs 839256 → 568408 (−32.27%). Runtime lưu ở hai run là 0.840 và 0.577 ms/img; chưa đủ bằng chứng khẳng định tăng tốc so với dense baseline. Taylor cùng ratio có accuracy 79.5% ở run trước và 83.5% ở run sau, nên cần kiểm tra khác biệt calibration/config trước khi xem là tương đương filter L1.

**SNIP** giữ accuracy 84% ở ratio 30–50%; **LAMP** đạt 85% ở ratio 50%; **SynFlow** đạt 85% ở ratio 70% trong các run đã lưu. Đây là ứng viên giữ accuracy khi tăng sparsity, nhưng dense parameters vẫn 64851, không chứng minh model file nhỏ hơn hoặc inference nhanh hơn. Chênh lệch 1 điểm % trên tập validation nhỏ, một seed, không chứng minh ưu thế thống kê. GraSP trong protocol hiện tại rơi về khoảng 1% accuracy: cần điều tra implementation/calibration/protocol, không kết luận GraSP nói chung kém hơn.

Trong nhóm B, channel/filter/Taylor không FT mất accuracy rất lớn; bỏ residual block giữ 52.4% so với 70% baseline, đổi lại giảm parameters 10.10%. Chưa có phương án cân bằng tốt được xác nhận. Trong nhóm C, filter L1 run chỉ nhắm layer được khai báo (`layer_params=[[0,0.1]]`, global_pruning=false dù tên run có “global”) giữ mAP tốt hơn nhưng giảm parameters rất ít và benchmark latency 19681.143 ms/img bất thường, không dùng để xếp hạng. Depth và Taylor mất mAP đáng kể. **Không có đủ dữ liệu để tuyên bố phương pháp tốt nhất về cả accuracy, model size và inference speed trên detection.**

L1/L0 regularization chỉ thấy cấu hình, chưa có kết quả hoàn chỉnh; attention-head pruning chưa có run tương ứng. Không gán số liệu paper khác vào ô còn thiếu. Để chọn phương pháp triển khai, cần đo dense/pruned cùng hardware và runtime, xác nhận GFLOPs cho mask, và thêm kết quả sau fine-tuning với cùng budget; giảm zero weights chỉ giúp tốc độ khi backend thực sự khai thác sparsity.

## Đối chiếu protocol với tài liệu gốc

Các nguồn này dùng để giải thích phạm vi phương pháp, **không trộn số liệu vào hai bảng thực nghiệm local**: [SNIP](https://arxiv.org/abs/1810.02340), [GraSP](https://arxiv.org/abs/2002.07376), [SynFlow](https://arxiv.org/abs/2006.05467), [L0 regularization](https://arxiv.org/abs/1712.01312), [filter pruning theo L1 norm](https://arxiv.org/abs/1608.08710), [Taylor pruning](https://arxiv.org/abs/1611.06440), [attention-head pruning bằng L0 gates](https://aclanthology.org/P19-1580/), [joint channel/layer/block pruning](https://arxiv.org/abs/2406.12079). Classification accuracy, BLEU, mAP 3D và COCO mAP không thể quy đổi lẫn nhau. PruningBench có [benchmark structural pruning chung trên YOLOv8/COCO](https://github.com/HollyLee2000/PruningBench), nhưng khác model/protocol local nên không dùng để bổ sung số liệu thiếu hoặc xếp hạng chung.

## Nguồn từng dòng

'''
md = intro + '## 1. Quality Benchmark\n\n' + table(qheaders,quality) + '\n\n## 2. Complexity & Runtime Benchmark\n\n' + table(rheaders,runtime) + '\n' + conclusion + '\n'.join(links) + '\n'
(OUT/'benchmark_report.md').write_text(md,encoding='utf-8')
(OUT/'benchmark_data.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')

def htable(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+html.escape(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'

def paragraphs(text):
    return ''.join('<p>'+html.escape(p).replace('\n','<br>')+'</p>' for p in text.strip().split('\n\n'))

page = '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pruning benchmark từ artifacts</title><style>body{font:15px/1.6 system-ui;margin:32px;color:#172033;background:#f8fafc}h1,h2{color:#0b4760}.scroll{overflow:auto;background:white;border:1px solid #cbd5e1;max-height:75vh}table{border-collapse:collapse;font-size:13px;min-width:1600px}th,td{padding:10px;border:1px solid #e2e8f0;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#123c50;color:white}tr:nth-child(even){background:#f1f5f9}p{max-width:1150px}a{color:#075985}</style><h1>Benchmark pruning từ artifacts</h1><p><a href="benchmark_report.md">Báo cáo Markdown và nguồn dẫn</a> · <a href="benchmark_data.json">Dữ liệu và cấu hình gốc</a></p>' + paragraphs(intro) + '<h2>1. Quality Benchmark</h2>' + htable(qheaders,quality) + '<h2>2. Complexity &amp; Runtime Benchmark</h2>' + htable(rheaders,runtime) + paragraphs(conclusion) + '<h2>Nguồn từng dòng</h2><ul>' + ''.join(f'<li>{x["id"]}: <a href="../artifacts/{html.escape(x["run"])}/result.json">{html.escape(x["run"])}</a></li>' for x in records) + '</ul></html>'
(OUT/'benchmark_report.html').write_text(page,encoding='utf-8')
print(f'Exported {len(records)} saved runs; {len(quality)} rows in each table to {OUT}')
