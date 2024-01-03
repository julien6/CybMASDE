import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import { Component, Input, TemplateRef, ViewChild } from '@angular/core';
import { LibraryElementComponent } from '../library-element/library-element.component';

@Component({
  selector: 'app-library',
  templateUrl: './library.component.html',
  styleUrls: ['./library.component.css']
})
export class LibraryComponent {

  centered = false;
  disabled = false;
  unbounded = false;

  libraryConfigs: any;

  constructor(public dialogRef: MatDialogRef<LibraryComponent>, public dialog: MatDialog) {
    this.getLibraryInCache().then((value) => {
      this.libraryConfigs = this.getLibraryConfigs();
    }).catch((err) => console.error(err))
  }

  // TODO : to replace by real REST service call
  getLibraryConfigs() {
    const config = {
      title: "Small-scaled company network under two attacks",
      short_description: "A simple small-scaled company network containing Linux and Windows machines splitted into four subnets.",
      md_file: "description.md",
      thumbnail_file: "thumbnail.png",
      data_file: "data.json"
    }

    const confS2 = JSON.parse(JSON.stringify(config))
    confS2["title"] = "An Adhoc drone swarm scenario"
    confS2["short_description"] = "A scenario with random red agent activation in a drone swarm"

    const confA1 = JSON.parse(JSON.stringify(config))
    confA1["title"] = "Random agent"
    confA1["short_description"] = "A simple agent that samples action randomly from action space"
    
    const confA2 = JSON.parse(JSON.stringify(config))
    confA2["title"] = "Decision tree"
    confA2["short_description"] = "A simple agent that samples action randomly from a decision tree"

    const confA3 = JSON.parse(JSON.stringify(config))
    confA3["title"] = "DQN Agent"
    confA3["short_description"] = "A simple agent that samples action randomly from a decision tree"

    const confN1 = JSON.parse(JSON.stringify(config))
    confN1["title"] = "Velociraptor"
    confN1["short_description"] = "Velociraptor is a tool for collecting host based state information"

    const confN2 = JSON.parse(JSON.stringify(config))
    confN2["title"] = "Kali Linux"
    confN2["short_description"] = "A Debian-derived Linux distribution designed for digital forensics and penetration testing"

    const confN3 = JSON.parse(JSON.stringify(config))
    confN3["title"] = "Ubuntu 18.03"
    confN3["short_description"] = "Ubuntu is a Linux distribution based on Debian and composed mostly of free and open-source software"

    const confN4 = JSON.parse(JSON.stringify(config))
    confN4["title"] = "Windows Server 2008"
    confN4["short_description"] = "A group of several proprietary graphical operating system families developed and marketed by Microsoft."

    return {
      Agents: { "A001": confA1, "A002": confA2, "A003": confA3 },
      Nodes: { "N001": confN1, "N002": confN2, "N003": confN3, "N004": confN4 },
      Scenarios: { "S001": config, "S002": confS2 }
    }
  }


  /* TODO: download the whole library in frontend cache asset folder from a REST service call */
  getLibraryInCache() {
    return new Promise<boolean>((resolve, reject) => {
      resolve(true);
    })
  }

  openLibraryElementDialog(elementType: string, elementId: string, descriptionFile: string) {
    this.dialog.open(LibraryElementComponent, {
      width: '100%',
      data: {
        title: this.libraryConfigs[elementType][elementId]["title"],
        md_file_path: '/assets/library/' + elementType.toLowerCase() + '/' + elementId + '/' + descriptionFile,
        data_file_path: '/assets/library/' + elementType.toLowerCase() + '/' + elementId + '/' + this.libraryConfigs[elementType][elementId]["data_file"]
      }
    });
  }

}
function $any(libraryIds: { Agents: string[]; Nodes: string[]; Scenarios: string[]; }) {
  throw new Error('Function not implemented.');
}

