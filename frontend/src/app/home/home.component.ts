import { Component, Optional } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { MenuBarComponent } from '../menu-bar/menu-bar.component';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent {

  recentProjects: any = [];

  rootUrl = "http://127.0.0.1:5000/";

  constructor(private http: HttpClient, @Optional() private menuBar: MenuBarComponent) { }

  ngOnInit(): void {
    this.loadRecentProjects();
  }

  // Fonction pour charger les projets récents depuis un fichier .txt
  loadRecentProjects() {
    this.http.get(this.rootUrl + 'get-recent-projects').subscribe(data => {
      this.recentProjects = data;
    }, error => {
      console.error('Erreur lors du chargement des projets récents', error);
    });
  }

  // Action pour créer un nouveau projet
  newProject() {
    this.menuBar.createNewProject();
  }

  // Action pour charger un projet
  loadProject() {
    this.menuBar.openProject();
  }

}
