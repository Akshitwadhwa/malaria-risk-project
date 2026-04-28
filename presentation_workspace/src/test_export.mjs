import {Presentation, PresentationFile} from '@oai/artifact-tool';
const p=Presentation.create({slideSize:{width:1920,height:1080}}); const s=p.slides.add();
s.background.fill = {type:'solid', color:'#ffffff'};
const sh=s.shapes.add({geometry:'rect', position:{left:100,top:100,width:500,height:100}, fill:{type:'solid', color:'#0f766e'}, line:{color:'#0f766e', transparency:100000}}); sh.text='Hello'; sh.text.fontSize=44; sh.text.color='#ffffff';
const ppt=await PresentationFile.exportPptx(p); await ppt.save('output/test.pptx');
const png=await s.export({format:'png'}); console.log('png type', typeof png, Object.keys(png||{}), png?.constructor?.name); console.log(png);
console.log('done')
