import { OnInit } from '@angular/core';
import { Component, TemplateRef, ViewChild } from '@angular/core';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import { MermaidAPI } from 'ngx-markdown';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { Inject } from '@angular/core';
import { LiveAnnouncer } from '@angular/cdk/a11y';
import { AfterViewInit } from '@angular/core';
import { MatSort, Sort, MatSortModule } from '@angular/material/sort';
import { MatTableDataSource, MatTableModule } from '@angular/material/table';


const ELEMENT_DATA: any[] = [
  { metric: "", description: "", value: "", constraint_respect: "", constraint: "" },
  { metric: "", description: "", value: "", constraint_respect: "", constraint: "" },
  { metric: "", description: "", value: "", constraint_respect: "", constraint: "" },
  { metric: "", description: "", value: "", constraint_respect: "", constraint: "" },
  { metric: "", description: "", value: "", constraint_respect: "", constraint: "" },
];

@Component({
  selector: 'app-global-matrixes',
  templateUrl: './global-matrixes.component.html',
  styleUrls: ['./global-matrixes.component.css']
})
export class GlobalMatrixesComponent implements OnInit {

  displayedColumns: string[] = ['metric', 'description', 'value', 'constraint_respect', 'constraint'];
  dataSource = new MatTableDataSource(ELEMENT_DATA);

  constructor(private _liveAnnouncer: LiveAnnouncer) { }

  @ViewChild(MatSort) sort: MatSort;

  ngAfterViewInit() {
    this.dataSource.sort = this.sort;
  }

  /** Announce the change in sort state for assistive technology. */
  announceSortChange(sortState: Sort) {
    // This example uses English messages. If your application supports
    // multiple language, you would internationalize these strings.
    // Furthermore, you can customize the message to add additional
    // details about the values being sorted.
    if (sortState.direction) {
      this._liveAnnouncer.announce(`Sorted ${sortState.direction}ending`);
    } else {
      this._liveAnnouncer.announce('Sorting cleared');
    }
  }

  options: MermaidAPI.Config = {
    fontFamily: '"trebuchet ms", verdana, arial, sans-serif',
    logLevel: MermaidAPI.LogLevel.Info,
    theme: MermaidAPI.Theme.Dark
  };

  data: any = `# Title`

  ngOnInit(): void {
  }

}
