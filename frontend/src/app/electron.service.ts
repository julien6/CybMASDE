import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class ElectronService {

  private ipcRenderer: any;
  private process

  constructor() {
    if (this.isElectron()) {
      this.ipcRenderer = window['electron'].ipcRenderer;
      this.process = window['electron'].process;
    } else {
      console.warn("L'application ne tourne pas dans Electron");
    }
  }

  // isElectron(): boolean {
  //   return !!(window && window.process && window.process.type) || !!(process && process.versions && process.versions['electron']);
  // }

  isElectron(): boolean {
    return !!(window && window['electron'] && window['electron'].process && window['electron'].process.versions);
  }

  async saveFile(): Promise<string | null> {
    return new Promise((resolve) => {
      if (this.ipcRenderer) {
        this.ipcRenderer.once('save-file-response', (event: any, filePath: any) => {
          resolve(filePath);
        });
        this.ipcRenderer.send('open-save-dialog');
      } else {
        resolve(null);
      }
    });
  }


}
