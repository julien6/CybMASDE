import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GlobalSeqdiagramsComponent } from './global-seqdiagrams.component';

describe('GlobalSeqdiagramsComponent', () => {
  let component: GlobalSeqdiagramsComponent;
  let fixture: ComponentFixture<GlobalSeqdiagramsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GlobalSeqdiagramsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GlobalSeqdiagramsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
