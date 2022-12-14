
import hashlib
import os,sys

from server_util import add_url_params

import flask
from datetime import datetime
#import flask
def str2hex(x,encoding='utf-8'):
    return (hashlib.md5(x.encode(encoding)).hexdigest())

from apiflask import APIFlask, Schema, abort, APIBlueprint
from apiflask.fields import Integer, String
from apiflask.validators import Length, OneOf

from flask import ( 
    url_for,render_template_string,
    render_template,
    redirect,
    request,
    )


import flask_wtf
from flask_wtf import Form
from wtforms import StringField
import wtforms as wtf
# [DOC] https://github.com/wtforms/flask-wtf/blob/main/src/flask_wtf/form.py
# [DOC] https://github.com/wtforms/wtforms/blob/3.0.x/src/wtforms/form.py
#DOC List of WTFfields https://wtforms.readthedocs.io/en/3.0.x/fields/



# app0 = APIFlask(__name__, static_folder='/shared_data/', static_url_path = '/shared_data/')
app0 = APIFlask(__name__, )
app = bp = APIBlueprint('root',__name__)

app0.config['PREFIX_GCAT'] = PREFIX_GCAT = prefix = '/gcat/'

# app0.config['NGL_APP_URL'] = NGL_APP_URL = 'http://localhost:9001/mdsrv/webapp/' 
app0.config['NGL_APP_URL'] = NGL_APP_URL = 'http://192.168.50.132:9001/mdsrv/webapp/' 

app0.config['APP_VERSION']='v0.0.1'

# @app0.route('/shared_data/', defaults={'req_path': ''})
# @app0.get('/shared_data/<path:req_path>',defaults={'req_path': ''})
@app0.get('/shared_data/<path:req_path>')
def dir_listing(req_path):
    BASE_DIR = '/shared_data/'
    req_path = req_path.rstrip('/')
    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)
    elif os.path.isfile(abs_path):
    # Check if path is a file and serve
        # return flask.send_file(abs_path)
        return flask.send_file(abs_path,as_attachment=False,mimetype='text/plain')
        #  as_attachment=True
    else:   
        # return [os.path.isfile(abs_path),abs_path]
        # print()
        # Show directory contents
        files = os.listdir(abs_path)
        return render_template_string(
            '''
    <ul>
        {% for file in files %}
        <li>
            <a href="{{ request.path + ['/',''][(request.path.endswith('/'))] + file }}">
                {{ request.path + ['/',''][(request.path.endswith('/'))] + file }}
            </a>
        </li>
        {% endfor %}
    </ul>        
            ''', files=files)



from collections import namedtuple
import glob
import json
from json import JSONDecoder, JSONDecodeError
import re
NOT_WHITESPACE = re.compile(r'[^\s]')
def json_decode_stacked(document, pos=0, decoder=JSONDecoder()):
    '''
    Source: https://stackoverflow.com/a/50384432/8083313
    '''
    while True:
        match = NOT_WHITESPACE.search(document, pos)        
        if not match:
            return
        pos = match.start()
        
        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj

@app.get('/')
def index_page():
    # [str(p) for p in temp_app.url_map.iter_rules()]
    return render_template_string('''
    <a href="{{PREFIX_GCAT}}"><h3>GCAT {{app.config.APP_VERSION}}</h3></a>
    <pre>
    debug:{{debug}}
    </pre>


    {% for x in app.url_map.iter_rules() %}

    <!--
    {{ x.__dict__.keys()     |safe }}
    <p>{{x}}</p>
    <p>{{x.__repr__()}}</p>
    <pre>{{x.__doc__}}</pre>
    -->


        {% if debug or ("GET" in x.methods and not x.arguments) %}
            {{ahref_from_url(x.rule) | safe}}</p>
            {% if debug %}
                <table border="1">
                <tbody>
                {% for k in "rule endpoint host methods arguments ".split() %}
                        <tr>
                            <td>
                            {{ k }}
                            </td>
                            <td>
                            {{ x.__dict__[k]     |safe }}
                            </td>raise
                        </tr>
                {% endfor %}
                </tbody></table>
            {% endif %}
        {% endif %}

    {% endfor %}

    ''',app=app0, debug=int(request.args.get('debug',0)), ahref_from_url=ahref_from_url)


PypeJsonData =namedtuple('PypeJsonData','path job_name shex msgs')
class PypeJsonData(PypeJsonData):
    def is_ended(self):
        return self.msgs[-1]['name'] == '_PYPE_END'

    @classmethod
    def from_file_path(cls,xx):
        path = xx
        xx = xx[len(SDATA_DIR)+1:]
        sp = xx.split('/',2)
        with open(path,'r') as f:
            it = json_decode_stacked(f.read())
            parsed = list(it)
        xxx = cls(path=path,job_name=sp[0],shex= sp[1], msgs = parsed,)        
        return xxx
if 1:
    '''
    PP_JSONS is the in-memory index of on-disk tasks
    PP_JSONS should be modified to reflect changes of 
    '''
    SDATA_DIR = '/shared_data'
    x = glob.glob(SDATA_DIR+'/*/*/PYPE.json')
    ret = []
    for xx in x:
        # xxx =  PypeJsonData.from_file_path(xx)
        ret.append(PypeJsonData.from_file_path(xx))
        # xxx)
    PP_JSONS = ret

ENV_RUNDIR = '/shared_data'
@app.route('/p_md_simple',methods='GET POST'.split())
def p_md_simple():
    IS_DEBUG_VIEW = int(request.args.get('IS_DEBUG_VIEW',False))
    rundir = request.args.get('rundir', ENV_RUNDIR)
    error = ""
    action_base, action_name = request.endpoint.rsplit('.',1)
    form = SimpleMdMessageForm(request.form)
    if request.method=='POST':
        ret = Actions.job_submit(action_name, form.data, rundir)
        ret_msg = render_template_string('''

    <table border="1">
    <thead></thead>
    <tbody>
    {% for k,v in ret.items() %}
            <tr>
                <td>
                {{ k }}
                </td>
                <td>
                {{ v |safe }}
                </td>
            </tr>
    {% endfor %}
    <tr>
        <td>
        output_data
        </td>
        <td>
            <table border="1">
            <tbody>
            {% for k,v in ret.get('output_data',{}).items() %}
                    <tr>
                        <td>
                        {{ k }}
                        </td>
                        <td>
                        {{ v |safe }}
                        </td>
                    </tr>
            {% endfor %}

            </tbody></table>
        </td>
    </tr>


    </tbody></table>
        ''',ret=ret,action_base=action_base,action_name=action_name)
    else:
        ret_msg = '<p>empty since NOT a POST request</p>'

    resp = render_template_string('''

<html>
<head><title>{{form.__class__.__name__}}</title>
</head>
    <h3>
    {{form.__class__.__name__}}
    </h3>
    {{IS_DEBUG_VIEW}}
    {{action_name}}
    {{request.view_args}}
    <pre>{{form.__doc__}}
    </pre>
    <table border="1">
    <thead></thead>
    <tbody>
    </tbody></table>


{# <form method="POST" action="{{action_base}}job_submit/{{form.job_name}}"> #}
<form method="POST" action="">

    {{ form.csrf_token }}


    <table border="1">
    <thead></thead>
    <tbody>
    {% for k,v in form._fields.items() %}
            <tr>
                <td>
            {{v.label}}
                </td>
                <td>
            {{v(size=20)}}
                </td>
            </tr>

    {% endfor %}

    </tbody></table>
    <input type="submit" value="Submit">

</form>

<br>
<h3>Returned Message</h3>

{{ ret_msg | safe }}
<br>


<h3>Loaded Results </h3>
    <table border="1">
    <thead></thead>
    <tbody>

    {% for v in PP_JSONS %}
      <tr>
      {% for f in "job_name shex path".split() %}
                <td>
                {{v[f]}}
                </td>
      {% endfor %}

      <td>
        {{ v.is_ended() }}
      </td>

      {% if IS_DEBUG_VIEW %}
      <td>
        {{ v.msgs[0:2] }}
      </td>
      <td>
        {{ v.msgs[-1:] }}
      </td>
      {% endif %}

    </tr>
    {% endfor %}

    </tbody>
    </table>

</html>
        ''',form=form,PP_JSONS=PP_JSONS, IS_DEBUG_VIEW=IS_DEBUG_VIEW,ret_msg=ret_msg,
        action_base = action_base,action_name=action_name)
    return resp


class SimpleMdMessage(Schema):
    '''
    `struct` is a path to a 
    '''
    struct = String(required=True, validate=Length(0, 150))
    md_steps = Integer()

class SimpleMdMessageForm(Form):
    '''
    Run a molecular simple dynamics simulation with gromacs
    Need to supply `struct`

    `TBC:DONE:impl a status checking page to show whether the long running process. probably with a term-like interface. or with PYPE.json` 
    `TBC:adding persistence mechanism to build index for existing tasks, probably with PYPE.json`


    `struct` is a path to a 
    '''
    struct = StringField()
    # required=True, validate=Length(0, 150))
    md_steps = wtf.IntegerField()
    # job_name = "md_simple"
    # md_steps = Integer()


class SimpleMdResult(Schema):
    result_pointer = String()

import json
import threading 


tdict = {}

@app.get('/job_status/<string:action_name>/<string:shex>')
# @app.input(SimpleMdMessage)
def job_status(action_name, shex,json=json):
    tkey = action_name + '/' + shex
    data = tdict[tkey]
    t = data['thread']
    is_alive = t.is_alive()
    datajson = json.dumps(data,indent=2,default=lambda x:x.__repr__())
    return flask.render_template_string('''
    <html>
        <body>
        <p>

        Job submitted!

        When the calculation is done, you can access it at 

        {{datajson}}
        </p>
        <pre>
        </pre>
        <table border="1">
        <thead></thead>
        <tbody>

            <tr>
                <td>
                    is_alive
                </td>
                <td>
                    {{is_alive}}
                </td>
            </tr>


            <tr>
                <td>
                    ts_start
                </td>
                <td>
                    {{data["ts_start"]}}
                </td>
            </tr>

            <tr>
                <td>
                    ret
                </td>
                <td>
                    {{data["ret"]}}
                </td>
            </tr>

        </tbody>
        </table>
        </body>        
        </html>''', **locals())    

    #  data):

@app.post('/job_submit/<string:action_name>')
@app.input(SimpleMdMessage,location='json_or_form')
def job_submit(action_name,data):
    """
    submit a job to a targeted
    """
    # data = request.data
    data0 = data.copy()
    is_ret_input = int(request.args.get('is_ret_input',0))
    if is_ret_input:
        return data0
    # print(data0)

    return Actions.job_submit(action_name, data)

    # tdict[tkey] = dict(thread=t, ts_start=datetime.now().isoformat(),ret=None,error=None)

    t.start()
    
    url_for_status = url_for( PREFIX_GCAT +".job_status",action_name=action_name, shex=data['shex'])
    # =tkey)
    dumps = json.dumps
    return flask.render_template_string('''
    <html>
        <body>
        <p>
        Job submitted!

        When the calculation is done, you can access it at 

        </p>
        <pre>
        </pre>
        <table border="1">
        <thead></thead>
        <tbody>

            <tr>
                <td>
                    recved data json
                </td>
                <td>
                    <pre>{{dumps(data0,indent=2)}}</pre>
                </td>
            </tr>


            <tr>
                <td>
                    job_status check url
                </td>
                <td>
                    <a href="{{url_for_status}}">{{url_for_status}}</a>
                    <iframe src="{{url_for_status}}"></iframe>
                </td>
            </tr>

        </tbody>
        </table>
        </body>
        
        </html>''', **locals())
        # locals())
        # data)



def ahref_from_url(url,extra=""):
    return f'<a href="{url}" {extra}>{url}</a>'
class Actions(object):
    @classmethod
    def job_submit(cls, action_name, data, rundir):
        '''
        If the job is already in cache, then just return the cache
        '''
        rundir = rundir + '/' + action_name
        act = getattr(cls,action_name)
        data['shex'] = shex = str2hex(json.dumps(sorted(data.items())))
        tkey = f'''{action_name}/{data['shex']}'''
        pype_json_file = f'''{rundir}/{data['shex']}/PYPE.json'''

        ret_data = dict(
            message = "Job submitted!", 
            data=data, 
            output_data = {},
            tkey=tkey,
            url_status = ahref_from_url(url_for(f'{PREFIX_GCAT}.job_status',action_name=action_name,shex=shex)),
            pype_json_file=ahref_from_url(pype_json_file)
            )
            
        check_only = False
        if os.path.exists(pype_json_file):
            pdata = PypeJsonData.from_file_path(pype_json_file) 
            if pdata.is_ended():
                ret_data['message'] = 'Job cache found and returned!'
                check_only = True

        def wrapper(data,tkey=tkey,act=act,rundir=rundir,check_only=check_only):
            try:


                tdict[tkey]['ret'] = ret_data['output_data'] = act( data, rundir, check_only)
                # xxx = PypeJsonData(path=path,job_name=sp[0],shex= sp[1], msgs = parsed,)        
                # ret.append(xxx) 
                PP_JSONS.append(PypeJsonData.from_file_path(pype_json_file))               
            except Exception as e:
                tdict[tkey]['error'] = e
                raise e

        if tdict.get(tkey,None) is not None and tdict[tkey]['thread'].is_alive():
            ret_data['message'] = 'Job thread found running. please wait and check job_status'
        else:
            t = threading.Thread(target=wrapper,args=(data,))
            tdict[tkey] = dict(input_data=data,thread=t)
            t.start()            
            if check_only:
                t.join()
        return ret_data



    @staticmethod
    def p_md_simple(data, rundir,check_only):
        """

        The sole idea is that the parameter decides which function to call

        Option 1: clone and proxy the request 
        - Pros
            - proxy is useful tech
            - And yes I am making another request to a internal 
        - Cons
            - information may be lost during proxying
            - the same request will be parsed multiple times
        - work: 5/10
        Option 2: Refactor to get callable from the registry 
        - Pros
            - fast
        - Cons
            - not sure 


        <http://localhost:9001/mdsrv/webapp/?struc=file://static-prot/1PGB/5123456/md_0_1.gro&traj=file://static-prot/1PGB/5123456/md_0_1.xtc&data=/mdsrv/&mdsrv=/mdsrv/>
        <http://localhost:9001/mdsrv/webapp/?struc=/mdsrv/file/static-prot/1PGB/5123456/md_0_1.gro&traj=file://static-prot/1PGB/5123456/md_0_1.xtc&data=/mdsrv/&mdsrv=/mdsrv/>

        """    
        param    = data.copy()
        struct   = data['struct']
        shex     = data['shex']
        md_steps = data['md_steps']
        # rundir   = '/shared_data/md_simple/'

        ### Param validation

        outdir = rundir+'/'+shex

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(f'{outdir}/PARAM.json','w') as f:
            json.dump(param,f,default=repr)

        if struct.startswith('file://'):
            struct = struct[len('file://'):]
            assert os.path.exists(struct),struct
            # assert 
            raise NotImplementedError('The pype recipe is not compatible with file:// url ')
        elif struct.startswith('rcsb://'):
            struct = struct[len('rcsb://'):]
        else:
            assert NotImplementedError(struct)
        
        from know_simple_md import Controller,know_simple_md
        from pype.controller import PypeExecResult
        if not check_only:
            pype2 = Controller.from_func(know_simple_md, 
                # GMX    = pype1.built["GMX"],
                GMX    = 'gmx',
                PDB_ID = struct,
                md_steps=md_steps,
                )    
            # outdir = rundir+'/'+struct + '/' + str(md_steps)
            pype2.run(rundir=outdir,extra_nodes=[
                    PypeExecResult(name='GCAT_PARAM',suc=1,extra_dict=param)

            ])


        data['shex'] = shex
        data['outdir'] = outdir
        reldir = outdir.lstrip('/')
        # fodir = 'file://'+outdir
        fodir = 'file://' + reldir
        # NGL_APP_URL = 'http://'
        # NGL_APP_URL = f'{request.base_url}/mdsrv/webapp/'
        # fudir = f'http://localhost:9001/mdsrv/file/{outdir}'

        # data['url_ngl'] = f'http://localhost:9001/mdsrv/webapp/?struc={fudir}/md_0_1.gro&traj={outdir.lstrip("/")}/md_0_1.xtc&data=/mdsrv/&mdsrv=/mdsrv/'
        # data['url_ngl'] = f'http://localhost:9001/mdsrv/webapp/?struc={fodir}/md_0_1.gro&traj={fodir}/md_0_1.xtc&data=/mdsrv/&file=/mdsrv/'
        url_ngl = add_url_params( 
            app0.config['NGL_APP_URL'],
            {
            "struc":f'{fodir}/md_0_1.gro',
            'traj':f'{fodir}/md_0_1.xtc',
            'data':'/mdsrv/',
            'file':'/mdsrv/'})

        data['url_ngl']  = url_ngl
        data['url_ngl_href']  = f'<a href="{url_ngl}">{url_ngl}</a>'
        data['rundir']   = rundir
        return data






app = app0
# Swagger UI
app.config['SWAGGER_UI_CSS'] = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.11.1/swagger-ui.min.css'
app.config['SWAGGER_UI_BUNDLE_JS'] = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.11.1/swagger-ui-bundle.min.js'
app.config['SWAGGER_UI_STANDALONE_PRESET_JS'] = 'https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.11.1/swagger-ui-standalone-preset.min.js'
# Redoc
app.config['REDOC_STANDALONE_JS'] = 'https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js'
# Elements
app.config['ELEMENTS_JS'] = 'https://cdn.jsdelivr.net/npm/@stoplight/elements-dev-portal@1.7.4/web-components.min.js'
app.config['ELEMENTS_CSS'] = 'https://cdn.jsdelivr.net/npm/@stoplight/elements-dev-portal@1.7.4/styles.min.css'
# RapiDoc
app.config['RAPIDOC_JS'] = 'https://cdn.jsdelivr.net/npm/rapidoc@9.3.2/dist/rapidoc-min.min.js'
# RapiPDF
app.config['RAPIPDF_JS'] = 'https://cdn.jsdelivr.net/npm/rapipdf@2.2.1/src/rapipdf.min.js'


app.register_blueprint(bp, url_prefix=prefix,name=prefix)
