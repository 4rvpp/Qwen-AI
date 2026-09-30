class Capture extends AudioWorkletProcessor {
  constructor() {
    super(); this.parts=[]; this.size=0; this.stopped=false;
    this.port.onmessage=e=>{if(e.data==='stop'){this.flush();this.stopped=true;this.port.postMessage({stopped:true});}};
  }
  flush(){
    if(!this.size)return;
    const frame=new Float32Array(this.size);let offset=0;
    for(const part of this.parts){frame.set(part,offset);offset+=part.length;}
    this.port.postMessage(frame.buffer,[frame.buffer]);this.parts=[];this.size=0;
  }
  process(inputs){
    const channel=inputs[0]?.[0];
    if(channel&&!this.stopped){this.parts.push(new Float32Array(channel));this.size+=channel.length;if(this.size>=sampleRate/4)this.flush();}
    return true;
  }
}
registerProcessor('capture',Capture);
