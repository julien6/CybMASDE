import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TreeComponent } from './tree-component.component';
import { MatTreeModule } from '@angular/material/tree';
import { MatIconModule } from '@angular/material/icon';
import { MatSliderModule } from '@angular/material/slider';
import { MatButtonModule } from '@angular/material/button';



@NgModule({
  declarations: [
    TreeComponent
  ],
  imports: [
    CommonModule,
    MatTreeModule,
    MatIconModule,
    MatSliderModule,
    MatButtonModule
  ],
  exports: [
    TreeComponent
  ]
})
export class TreeModule { }
