import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AgentsSeqdiagramsComponent } from './agents-seqdiagrams.component';

describe('AgentsSeqdiagramsComponent', () => {
  let component: AgentsSeqdiagramsComponent;
  let fixture: ComponentFixture<AgentsSeqdiagramsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AgentsSeqdiagramsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AgentsSeqdiagramsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
