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
                               (u.indexOf('features.json')>=0?analysis:payload))});
    w.HTMLCanvasElement.prototype.getContext=function(){return{
      // the whole 2D surface a page might touch: a stub missing one method
      // fails the page for a reason a browser would never have
      drawImage(){},putImageData(){},fillRect(){},clearRect(){},strokeRect(){},
      beginPath(){},closePath(){},moveTo(){},lineTo(){},arc(){},arcTo(){},
      ellipse(){},rect(){},bezierCurveTo(){},quadraticCurveTo(){},
      fill(){},stroke(){},clip(){},save(){},restore(){},translate(){},
      rotate(){},scale(){},setTransform(){},transform(){},
      fillText(){},strokeText(){},setLineDash(){},
      measureText(){return{width:0};},
      createLinearGradient(){return{addColorStop(){}};},
      globalAlpha:1,fillStyle:'',strokeStyle:'',lineWidth:1,lineCap:'',
      lineJoin:'',font:'',textAlign:'',textBaseline:'',
      getImageData:(a,b,c,d)=>({data:new w.Uint8ClampedArray(c*d*4)}),
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
  if (!problems.length && visible < 1)
    console.log('      rendered nothing into #body');
  process.exit(failed?1:0);
},900);