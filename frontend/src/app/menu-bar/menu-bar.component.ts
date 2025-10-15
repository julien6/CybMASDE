import { Component, ElementRef, ViewChild, ChangeDetectorRef } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ElectronService } from '../electron.service';
import { AboutDialogComponent } from '../about-dialog/about-dialog.component';
import { MatDialog } from '@angular/material/dialog';
import { ConfigEditorService } from '../config-editor.service';

@Component({
  selector: 'app-menu-bar',
  templateUrl: './menu-bar.component.html',
  styleUrls: ['./menu-bar.component.css']
})
export class MenuBarComponent {

  @ViewChild('fileInput', { static: false })
  fileInput!: ElementRef;

  constructor(public configEditorService: ConfigEditorService, private cdr: ChangeDetectorRef, private dialog: MatDialog) {
    this.configEditorService.config$.subscribe(config => {
      if (config !== null) {
        // Do something with the config
      }
    });
  }

  ngOnInit() {
  }

  openProject() {
    this.configEditorService.openProjectDialog().then((config) => {
      // Handle the loaded project configuration
      console.log('Loaded project configuration:', config);
      this.cdr.detectChanges();
    }).catch((error) => {
      console.error('Error while loading project:', error);
    });
  }

  openAbout() {
    this.dialog.open(AboutDialogComponent, {
      width: '500px'
    });
  }

  closeApp() {
    if (window.electron && window["electron"].closeApp !== null) {
      window["electron"].closeApp();
    } else {
      console.warn("La fonction closeApp n'est pas disponible.");
    }
  }

}
