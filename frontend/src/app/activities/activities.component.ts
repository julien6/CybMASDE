import { Component, OnInit, inject } from '@angular/core';
import { ProjectConfig } from '../models/config.model';
import { ConfigEditorService } from '../config-editor.service';
import { default_config } from '../models/default_project_configuration';

@Component({
  selector: 'app-activities',
  templateUrl: './activities.component.html',
  styleUrls: ['./activities.component.css']
})
export class ActivitiesComponent implements OnInit {

  public config!: ProjectConfig;

  constructor(private configEditorService: ConfigEditorService) {
    this.configEditorService.config$.subscribe(config => {
      if (config) { this.config = config; }
      else { this.config = default_config as ProjectConfig; }
    });
  }

  ngOnInit() {
  }

}
