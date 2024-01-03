import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Mpld3Component } from './mpld3.component';


@NgModule({
  declarations: [
    Mpld3Component
  ],
  imports: [
    CommonModule
  ],
  exports: [
    Mpld3Component
  ]
})
export class Mpld3Module { }
