import { ChangeDetectorRef, Component, Optional } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MenuBarComponent } from '../menu-bar/menu-bar.component';
import { ConfigEditorService } from '../config-editor.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {

  constructor(public configEditorService: ConfigEditorService, private cdr: ChangeDetectorRef) { }

  ngOnInit(): void {
    this.loadRecentProjects();
  }

  openProjectPath(path: string) {
    this.configEditorService.loadProject(path + "/project_configuration.json").then((config) => {
      // Handle the loaded project configuration
      console.log('Loaded project configuration:', config);
      this.cdr.detectChanges();
    }).catch((error) => {
      console.error('Error while loading project:', error);
    });
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

  recentProjects: any = [];

  // Fonction pour charger les projets récents depuis un fichier .txt
  loadRecentProjects() {
    this.configEditorService.getRecentProjects().then((data) => {
      this.recentProjects = data;
    }).catch((error) => {
      console.error('Erreur lors du chargement des projets récents', error);
    });
  }

}
