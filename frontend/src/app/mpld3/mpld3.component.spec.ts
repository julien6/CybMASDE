import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Mpld3Component } from './mpld3.component';

describe('Mpld3Component', () => {
  let component: Mpld3Component;
  let fixture: ComponentFixture<Mpld3Component>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ Mpld3Component ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Mpld3Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
