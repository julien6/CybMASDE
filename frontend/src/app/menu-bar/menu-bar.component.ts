import { Component, TemplateRef, ViewChild } from '@angular/core';
import { MatMenuTrigger } from '@angular/material/menu';
import { MatDialog } from '@angular/material/dialog';
import { PreferencesComponent } from '../preferences/preferences.component';
import { LibraryComponent } from '../library/library.component';
import { HistoryComponent } from '../history/history.component';
import { PropertiesComponent } from '../properties/properties.component';
import { ConfigureRunComponent } from '../configure-run/configure-run.component';
import { PackagingComponent } from '../packaging/packaging.component';
import { ConfigShareService } from '../services/config-share.service';
import { ServerAPIService } from '../services/serverAPI.service';
import { DataService } from "../services/app.data.service"

@Component({
  selector: 'app-menu-bar',
  templateUrl: './menu-bar.component.html',
  styleUrls: ['./menu-bar.component.css']
})
export class MenuBarComponent {

  constructor(private serverAPIService: ServerAPIService, private dataService: DataService, public dialog: MatDialog, private configShareService: ConfigShareService) { }

  ngOnInit() {
  }

  @ViewChild(MatMenuTrigger) trigger!: MatMenuTrigger;

  disable_properties: boolean = true;
  disable_history: boolean = true;
  disable_scenario_creation: boolean = false;
  disable_node_creation: boolean = true;
  disable_agent_creation: boolean = true;

  new_scenario() {
    this.disable_scenario_creation = true;
    this.disable_node_creation = false;
    this.disable_agent_creation = false;
  }

  new_agent() {

  }

  new_node() {

  }

  shown_toolbars: boolean = true;
  shown_environment_properties_panel: boolean = true;
  shown_graphical_visualization_panel: boolean = true;
  shown_prompt_interface: boolean = true;

  show_hide_toolbars() {
    this.shown_toolbars = !this.shown_toolbars;
    if (!this.shown_toolbars) {
      // hide toolbars
    } else {
      // show toolbars
    }
  }

  show_environment_properties_panel() {
    this.shown_environment_properties_panel = !this.shown_environment_properties_panel;
    if (!this.shown_environment_properties_panel) {
      // hide environment properties panel
      this.configShareService.hideShow("properties", false);
    } else {
      // show environment properties panel
      this.configShareService.hideShow("properties", true);
    }
  }

  show_graphical_visualization_panel() {
    this.shown_graphical_visualization_panel = !this.shown_graphical_visualization_panel;
    if (!this.shown_graphical_visualization_panel) {
      // hide graphical visualization panel
    } else {
      // show graphical visualization panel
    }
  }

  show_prompt_interface() {
    this.shown_prompt_interface = !this.shown_prompt_interface;
    if (!this.shown_prompt_interface) {
      // hide prompt interface
    } else {
      // show prompt interface
    }
  }

  openDialogWithRef(ref: TemplateRef<any>, height_percentage: string = "auto", width_percentage: string = "auto") {
    this.dialog.open(ref, {
      height: height_percentage,
      width: width_percentage
    });
  }

  openPreferencesDialog() {
    this.dialog.open(PreferencesComponent, {
      width: '100%'
    });
  }

  openLibraryDialog() {
    this.dialog.open(LibraryComponent, {
      width: '100%'
    });
  }

  openHistoryDialog() {
    this.dialog.open(HistoryComponent, {
      width: '100%'
    });
  }

  openPropertiesDialog() {
    this.dialog.open(PropertiesComponent, {
      width: '100%'
    });
  }

  openEmSimDialog(mode: string) {
    this.dialog.open(ConfigureRunComponent, {
      width: '100%',
      data: { "running_mode": mode }
    });
  }

  openPackagingDialog() {
    this.dialog.open(PackagingComponent, {
      width: '100%'
    });
  }

  menuMethod() {
    this.trigger.openMenu();
  }

  run() {
    throw new Error('Method not implemented.');
  }

  pause() {
    throw new Error('Method not implemented.');
  }

  next(episode_number: boolean = true, iteration_number: boolean = true, observation_spaces: boolean = true, agents_actions: boolean = true,
    agents_observations: boolean = true, agents_rewards: boolean = true, true_states: boolean = true, network_graph: boolean = true,
    team_agent_mapping: boolean = true, action_spaces: boolean = true, pz_cyborg_actions: boolean = true, cyborg_actions: boolean = true,
    cyborg_observations: boolean = true, agents_cyborg_comms: boolean = true, agents_pz_comms: boolean = true) {
    this.serverAPIService.getRequest("simulation-next?episode_number=" + episode_number + "&iteration_number=" + iteration_number
      + "&observation_spaces=" + observation_spaces + "&agents_actions=" + agents_actions + "&agents_observations=" + agents_observations
      + "&agents_rewards=" + agents_rewards + "&true_states=" + true_states + "&network_graph=" + network_graph +
      "&team_agent_mapping=" + team_agent_mapping + "&action_spaces=" + action_spaces + "&pz_cyborg_actions=" + pz_cyborg_actions
      + "&cyborg_actions=" + cyborg_actions + "&cyborg_observations=" + cyborg_observations + "&agents_cyborg_comms=" + agents_cyborg_comms
      + "&agents_pz_comms=" + agents_pz_comms).subscribe((data) => {
        this.dataService.setStateData(data)
      })
  }

  stop() {
    throw new Error('Method not implemented.');
  }

  runPackaging() {
    throw new Error('Method not implemented.');
  }

  resume() {
    throw new Error('Method not implemented.');
  }

}
