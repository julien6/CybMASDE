import { Component, OnInit, inject } from '@angular/core';
import {FormBuilder, Validators, FormsModule, ReactiveFormsModule} from '@angular/forms';

@Component({
  selector: 'app-transfering',
  templateUrl: './transfering.component.html',
  styleUrls: ['./transfering.component.css']
})
export class TransferingComponent implements OnInit {

  private _formBuilder = inject(FormBuilder);

  transferingFormGroup = this._formBuilder.group({
    transferingCtrl: ['', Validators.required],
  });

  constructor() { }

  ngOnInit() {
  }

}
