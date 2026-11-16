// Render service/static/research.html to docs/research/RuseTect research review 2026-11.pdf (A4 landscape).
// Usage: node research/make_pdf.js   (after python3 -I research/build_report.py)
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const root = path.resolve(__dirname, '..');
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + root + '/service/static/research.html', { waitUntil: 'networkidle' }).catch(() => {});
  await p.addStyleTag({ content: `nav.site{display:none} body{padding:0!important} .tbl,.chart{overflow:visible!important}
    .wrap table{min-width:0!important} table{font-size:11.5px} th,td{padding:6px 8px} main{gap:34px}
    .card,.chart,tr,.kpis{break-inside:avoid} h2,h3,.eyebrow{break-after:avoid}` });
  await p.pdf({ path: path.join(root, 'docs/research/RuseTect research review 2026-11.pdf'), format: 'A4', landscape: true,
    printBackground: true, margin: { top: '14mm', bottom: '14mm', left: '12mm', right: '12mm' }, displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font-size:8px;width:100%;text-align:center;color:#888">RuseTect research review · page <span class="pageNumber"></span> of <span class="totalPages"></span></div>' });
  await b.close();
})();
