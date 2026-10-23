/* Load one page in a real DOM and report what a browser would have reported.
 *
 * Checking each script on its own catches a typo and nothing else. The faults
 * that actually reach people are the ones that need the whole page running: a
 * name declared in two scripts that share a scope, or a control referenced by
 * an id that is not in the markup. Both leave a blank screen with the reason
 * only in the console.
 *
 *   node tests/render_page.js summary
 *
 * Exits non-zero if the page reports an error or renders nothing.
 */
const fs=require('fs'), path=require('path');
const {JSDOM, VirtualConsole}=require('jsdom');
const DIR=path.join(__dirname, '..', 'cichlid_bower_claude',
                    'server', 'templates') + path.sep;
const page=process.argv[2]||'summary';

// --- payloads the page will ask for -----------------------------------
const ND=10, N=6000, BIDS=['c','p','b','f','t','m','s','d','o','x'];
function enc(a,T){return Buffer.from(new T(a).buffer).toString('base64');}
const x=[],y=[],bid=[],prob=[],flags=[],day=[],trial=[],hour=[];
for(let i=0;i<N;i++){const d=i%ND;
  x.push(200+((i*37)%900)); y.push(150+((i*53)%700));
  bid.push(BIDS.indexOf(['c','p','b','f','t','m','s','o','x'][i%9]));
  prob.push(200); flags.push(3); day.push(d);
  trial.push(d<5?1:2); hour.push(8+(i%10));}
const packed={n:N,bids:BIDS,labels:{},x:enc(x,Uint16Array),y:enc(y,Uint16Array),
  bid:enc(bid,Uint8Array),prob:enc(prob,Uint8Array),flags:enc(flags,Uint8Array),
  day:enc(day,Uint16Array),trial:enc(trial,Uint8Array),hour:enc(hour,Uint8Array),
  summary:{n:N,noClip:120,unclustered:90,clusteredFraction:0.985}};
const png=u=>({url:u,width:64,height:48,scale:0.01,offset:0});
const days=[...Array(ND)].map((_,i)=>({index:i,trial:i<5?1:2,
  date:'2025-08-0'+((i%9)+1),hours:9.6,
  first:'2025-08-0'+((i%9)+1)+'T08:00:00',
  last:'2025-08-0'+((i%9)+1)+'T17:55:00',
  firstPng:png('f'+i),lastPng:png('l'+i),residualPng:png('r'+i),
  firstJpg:'j'+i+'.jpg',videoJpg:'v'+i+'.jpg',
  residualStats:{median:0.02,madFactor:1.2,n:900}}));
const payload={projectID:'MC_920_t001_tr1',analysisID:'YH_MC_Parentals',
  tankID:'t001',depthSize:[640,480],videoSize:[1296,972],downsample:2,
  pixelLength:0.103,days,trials:[{number:1},{number:2}],
  candidates:[{trial:1,kind:'start',offset:30,stem:'s',depthPng:png('c'),jpg:'c.jpg',
               index:5,time:'2025-08-01 08:30',label:'start +30 min'}],
  pairs:[{trial:1,side:'first',gapMinutes:0.2,piJpg:'pi.jpg',depthJpg:'d.jpg'},
         {trial:2,side:'first',gapMinutes:0.3,piJpg:'pi2.jpg',depthJpg:'d2.jpg'}],
  hasClusters:true,offsets:[0,30,60],residualScore:png('score'),
  logIssues:[],missing:[],
  prep:{trials:{},residual_k:5,day_marks:{},
        transform:[[0.49,0.01,-60],[0.005,0.49,-50],[0,0,1]],
        depth_crop:[[20,15],[600,15],[600,460],[20,460]],
        video_crop:[[150,120],[1180,130],[1190,900],[140,890]],overrides:{}}};
const analysis={analysisID:'A',threshold:0.6,coverage:0.68,projects:[
  {projectID:'P1',category:'MC',depthSpan:8,depthBins:64,trials:[
    {trial:1,excluded:false,spawnDepthHistogram:new Array(64).fill(3),
     spawnsMeasured:192,spawnDepthMedian:1.5,spawnOverCastle:80}]}]};

let lastDrawn='';
// one flat row per trial, as /<analysis>/trials.json returns them
const CATS=['MC','TI','CVxMC_F1'];
const trialRows={analysisID:'A',
  params:{confidence:0.67,hour_from:8,hour_to:18,threshold:0.6,coverage:0.68,
          split_at:0.5,min_for_shape:25,spread:30},
  skipped:[{projectID:'P9',why:'not collected'}],
  rows:[...Array(24)].map((_,i)=>({
    projectID:'P'+(1+(i%8)), tankID:'t'+(i%8), category:CATS[i%3], trial:1+(i%2),
    days:20+i, totalVolume:300+17*i, castleVolume:150+11*i, pitVolume:120+6*i,
    bowerIndex:-0.6+0.05*i, volumePerDay:12+0.4*i, grossVolume:800+30*i,
    nightShare:0.1+0.03*(i%9), nightRateRatio:0.4+0.07*(i%7),
    firstHalfIndex:-0.4+0.04*i, secondHalfIndex:0.2-0.03*i,
    events:4000+90*i, eventsPerDay:180+4*i, scoops:900+13*i, spits:880+15*i,
    spawns:40+i, scoopSpitRatio:0.8+0.02*(i%11),
    scoopArea:60+4*i, spitArea:70+3*i, scoopAspect:1.1+0.05*(i%8),
    spitAspect:1.2+0.04*(i%6), scoopNullAspect:1.07, spitNullAspect:1.07,
    centroidGap:1+0.2*(i%9), spawnDepthMedian:-0.8+0.1*(i%12),
    spawnOverCastle:10+3*(i%14), spawnsMeasured:30+i,
    newBowerDays:i%4, wallBuildingDays:i%3, daysTo90:3+(i%9)}))};

const problems=[];
const vc=new VirtualConsole();
vc.on('jsdomError', e => problems.push('ERROR  ' + (e.detail||e.message)));
vc.on('error', (...a) => problems.push('console.error  ' + a.join(' ')));

const html=fs.readFileSync(DIR+page+'.html','utf8').replace('__TITLE__','t')
            .replace('__ANALYSIS__','A').replace('__ROWS__','').replace('__COUNT__','0')
            .replace('__DONE__','0');
// inline the scripts so they run as real tags sharing one global scope,
// which is what a browser does and what separate eval() calls do not
const inlined=html.replace(/<script src="([^"]+)"><\/script>/g,
  (m, src) => '<script>' + fs.readFileSync(DIR+src,'utf8') + '<\/script>');
const dom=new JSDOM(inlined,{runScripts:'dangerously',virtualConsole:vc,
  url:'http://localhost:8080/p/'+page,
  beforeParse(w){
    w.fetch=(u)=>Promise.resolve({ok:true,status:200,
      json:()=>Promise.resolve(u.indexOf('clusters.json')>=0?packed:
                               (u.indexOf('features.json')>=0?analysis:
                               (u.indexOf('trials.json')>=0?trialRows:payload)))});
    w.HTMLCanvasElement.prototype.getContext=function(){return{
      // the whole 2D surface a page might touch: a stub missing one method
      // fails the page for a reason a browser would never have
      drawImage(img){ lastDrawn = (img && img.src) || ''; },
      putImageData(){},fillRect(){},clearRect(){},strokeRect(){},
      beginPath(){},closePath(){},moveTo(){},lineTo(){},arc(){},arcTo(){},
      ellipse(){},rect(){},bezierCurveTo(){},quadraticCurveTo(){},
      fill(){},stroke(){},clip(){},save(){},restore(){},translate(){},
      rotate(){},scale(){},setTransform(){},transform(){},
      fillText(){},strokeText(){},setLineDash(){},
      measureText(){return{width:0};},
      createLinearGradient(){return{addColorStop(){}};},
      globalAlpha:1,fillStyle:'',strokeStyle:'',lineWidth:1,lineCap:'',
      lineJoin:'',font:'',textAlign:'',textBaseline:'',
      // A depth PNG that decodes to NaN everywhere leaves every map empty, so
      // the arithmetic the pages exist for never runs and the test only proves
      // they can render nothing. Synthesise a surface instead, keyed on which
      // image was drawn, so that successive endpoints differ and a difference
      // between them is not identically zero.
      getImageData:(a,b,c,d)=>{
        const out=new w.Uint8ClampedArray(c*d*4);
        let seed=0;
        for(let i=0;i<lastDrawn.length;i++) seed=(seed*31+lastDrawn.charCodeAt(i))%997;
        for(let i=0,pi=0;i<c*d;i++,pi+=4){
          const row=Math.floor(i/c), col=i%c;
          // a mound that grows with the seed, over a flat base, in the units
          // loadDepth expects: (R<<8|G)*scale + offset
          const bump=Math.exp(-(((col-c*0.6)/(c*0.2))**2+((row-d*0.5)/(d*0.25))**2));
          const value=Math.round(400+seed*0.12+bump*seed*0.5);
          out[pi]=(value>>8)&255; out[pi+1]=value&255; out[pi+2]=255; out[pi+3]=255;
        }
        return {data:out};
      },
      createImageData:(c,d)=>({data:new w.Uint8ClampedArray(c*d*4)})};};
    // jsdom fetches no images, so a page that waits for one would look blank
    // for a reason a browser would never have
    const RealImage = w.Image;
    w.Image = function(){
      const node = new RealImage();
      let src;
      Object.defineProperty(node, 'src', {
        get(){ return src; },
        set(v){ src = v; setTimeout(()=>{ if (node.onload) node.onload();
                  node.dispatchEvent(new w.Event('load')); }, 0); }});
      Object.defineProperty(node, 'complete', { get(){ return !!src; } });
      return node;
    };
    w.addEventListener('error', e => problems.push('onerror  ' + e.message));
  }});
const w=dom.window;

setTimeout(()=>{
  const body=w.document.getElementById('body');
  const visible=body?body.children.length:-1;
  const text=(body?body.textContent:'').trim().slice(0,140);
  const failed = problems.length > 0 || visible < 1;
  console.log((failed?'FAIL  ':'ok    ')+page+
              '  ('+visible+' nodes) '+(text?text.slice(0,60):'(empty)'));
  problems.slice(0,6).forEach(p=>console.log('      '+p));
  // the whole rendered text, so a caller can assert a panel really appeared
  // rather than only that nothing threw
  // the whole document, not just #body: the controls bar, the meta strip and
  // the title sit outside it, and a control that failed to render there is
  // exactly as broken as one missing from the body
  if (process.argv[3] === '--text') {
    // a clone with the scripts taken out: the harness inlines them as real
    // tags, and textContent on the body would otherwise return every line of
    // the page's own source as though it were rendered text
    let shown = '';
    if (w.document.body) {
      const clone = w.document.body.cloneNode(true);
      clone.querySelectorAll('script,style').forEach(n => n.remove());
      shown = clone.textContent.replace(/\s+/g,' ').trim();
    }
    console.log('TEXT  ' + shown);
  }
  if (!problems.length && visible < 1)
    console.log('      rendered nothing into #body');
  process.exit(failed?1:0);
},900);
